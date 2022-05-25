# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class SABLHead(BaseModule):
    """Side-Aware Boundary Localization (SABL) for RoI-Head.

    Side-Aware features are extracted by conv layers
    with an attention mechanism.
    Boundary Localization with Bucketing and Bucketing Guided Rescoring
    are implemented in BucketingBBoxCoder.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        cls_in_channels (int): Input channels of cls RoI feature. \
            Defaults to 256.
        reg_in_channels (int): Input channels of reg RoI feature. \
            Defaults to 256.
        roi_feat_size (int): Size of RoI features. Defaults to 7.
        reg_feat_up_ratio (int): Upsample ratio of reg features. \
            Defaults to 2.
        reg_pre_kernel (int): Kernel of 2D conv layers before \
            attention pooling. Defaults to 3.
        reg_post_kernel (int): Kernel of 1D conv layers after \
            attention pooling. Defaults to 3.
        reg_pre_num (int): Number of pre convs. Defaults to 2.
        reg_post_num (int): Number of post convs. Defaults to 1.
        num_classes (int): Number of classes in dataset. Defaults to 80.
        cls_out_channels (int): Hidden channels in cls fcs. Defaults to 1024.
        reg_offset_out_channels (int): Hidden and output channel \
            of reg offset branch. Defaults to 256.
        reg_cls_out_channels (int): Hidden and output channel \
            of reg cls branch. Defaults to 256.
        num_cls_fcs (int): Number of fcs for cls branch. Defaults to 1.
        num_reg_fcs (int): Number of fcs for reg branch.. Defaults to 0.
        reg_class_agnostic (bool): Class agnostic regression or not. \
            Defaults to True.
        norm_cfg (dict): Config of norm layers. Defaults to None.
        bbox_coder (dict): Config of bbox coder. Defaults 'BucketingBBoxCoder'.
        loss_cls (dict): Config of classification loss.
        loss_bbox_cls (dict): Config of classification loss for bbox branch.
        loss_bbox_reg (dict): Config of regression loss for bbox branch.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_classes,
                 cls_in_channels=256,
                 reg_in_channels=256,
                 roi_feat_size=7,
                 reg_feat_up_ratio=2,
                 reg_pre_kernel=3,
                 reg_post_kernel=3,
                 reg_pre_num=2,
                 reg_post_num=1,
                 cls_out_channels=1024,
                 reg_offset_out_channels=256,
                 reg_cls_out_channels=256,
                 num_cls_fcs=1,
                 num_reg_fcs=0,
                 reg_class_agnostic=True,
                 norm_cfg=None,
                 bbox_coder=dict(
                     type='BucketingBBoxCoder',
                     num_buckets=14,
                     scale_factor=1.7),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox_reg=dict(
                     type='SmoothL1Loss', beta=0.1, loss_weight=1.0),
                 init_cfg=None):
        super(SABLHead, self).__init__(init_cfg)
        self.cls_in_channels = cls_in_channels
        self.reg_in_channels = reg_in_channels
        self.roi_feat_size = roi_feat_size
        self.reg_feat_up_ratio = int(reg_feat_up_ratio)
        self.num_buckets = bbox_coder['num_buckets']
        assert self.reg_feat_up_ratio // 2 >= 1
        self.up_reg_feat_size = roi_feat_size * self.reg_feat_up_ratio
        assert self.up_reg_feat_size == bbox_coder['num_buckets']
        self.reg_pre_kernel = reg_pre_kernel
        self.reg_post_kernel = reg_post_kernel
        self.reg_pre_num = reg_pre_num
        self.reg_post_num = reg_post_num
        self.num_classes = num_classes
        self.cls_out_channels = cls_out_channels
        self.reg_offset_out_channels = reg_offset_out_channels
        self.reg_cls_out_channels = reg_cls_out_channels
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.reg_class_agnostic = reg_class_agnostic
        assert self.reg_class_agnostic
        self.norm_cfg = norm_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox_reg = build_loss(loss_bbox_reg)

        self.cls_fcs = self._add_fc_branch(self.num_cls_fcs,
                                           self.cls_in_channels,
                                           self.roi_feat_size,
                                           self.cls_out_channels)

        self.side_num = int(np.ceil(self.num_buckets / 2))

        if self.reg_feat_up_ratio > 1:
            self.upsample_x = nn.ConvTranspose1d(
                reg_in_channels,
                reg_in_channels,
                self.reg_feat_up_ratio,
                stride=self.reg_feat_up_ratio)
            self.upsample_y = nn.ConvTranspose1d(
                reg_in_channels,
                reg_in_channels,
                self.reg_feat_up_ratio,
                stride=self.reg_feat_up_ratio)

        self.reg_pre_convs = nn.ModuleList()
        for i in range(self.reg_pre_num):
            reg_pre_conv = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=reg_pre_kernel,
                padding=reg_pre_kernel // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_pre_convs.append(reg_pre_conv)

        self.reg_post_conv_xs = nn.ModuleList()
        for i in range(self.reg_post_num):
            reg_post_conv_x = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=(1, reg_post_kernel),
                padding=(0, reg_post_kernel // 2),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_post_conv_xs.append(reg_post_conv_x)
        self.reg_post_conv_ys = nn.ModuleList()
        for i in range(self.reg_post_num):
            reg_post_conv_y = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=(reg_post_kernel, 1),
                padding=(reg_post_kernel // 2, 0),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_post_conv_ys.append(reg_post_conv_y)

        self.reg_conv_att_x = nn.Conv2d(reg_in_channels, 1, 1)
        self.reg_conv_att_y = nn.Conv2d(reg_in_channels, 1, 1)

        self.fc_cls = nn.Linear(self.cls_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

        self.reg_cls_fcs = self._add_fc_branch(self.num_reg_fcs,
                                               self.reg_in_channels, 1,
                                               self.reg_cls_out_channels)
        self.reg_offset_fcs = self._add_fc_branch(self.num_reg_fcs,
                                                  self.reg_in_channels, 1,
                                                  self.reg_offset_out_channels)
        self.fc_reg_cls = nn.Linear(self.reg_cls_out_channels, 1)
        self.fc_reg_offset = nn.Linear(self.reg_offset_out_channels, 1)

        if init_cfg is None:
            self.init_cfg = [
                dict(
                    type='Xavier',
                    layer='Linear',
                    distribution='uniform',
                    override=[
                        dict(type='Normal', name='reg_conv_att_x', std=0.01),
                        dict(type='Normal', name='reg_conv_att_y', std=0.01),
                        dict(type='Normal', name=