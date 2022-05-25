# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d, MaskedConv2d
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, calc_region,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


class FeatureAdaption(BaseModule):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deform conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deform_groups (int): Deformable conv group size.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.1,
                     override=dict(
                         type='Normal', name='conv_adaption', std=0.01))):
        super(FeatureAdaption, self).__init__(init_cfg)
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module()
class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.

    - Sampled 9 pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on. (squares)
    - Guided anchors.

    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
        approx_anchor_generator (dict): Config dict for approx generator
        square_anchor_generator (dict): Config dict for square generator
        anchor_coder (dict): Config dict for anchor coder
        bbox_coder (dict): Config dict for bbox coder
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        deform_groups: (int): Group number of DCN in
            FeatureAdaption module.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
        loss_loc (dict): Config of location loss.
        loss_shape (dict): Config of anchor shape loss.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox regression loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            approx_anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=8,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            square_anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                scales=[8],
                strides=[4, 8, 16, 32, 64]),
            anchor_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            reg_decoded_bbox=False,
            deform_groups=4,
            loc_filter_thr=0.01,
            train_cfg=None,
            test_cfg=None,
            loss_loc=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                           loss_weight=1.0),
            init_cfg=dict(type='Normal', layer='Conv2d', std=0.01,
                          override=dict(type='Normal',
                                        name='conv_loc',
                                        std=0.01,
                                        bias_prob=0.01))):  # yapf: disable
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.deform_groups = deform_groups
        self.loc_filter_thr = loc_filter_thr

        # build approx_anchor_generator and square_anchor_generator
        assert (approx_anchor_generator['octave_base_scale'] ==
                square_anchor_generator['scales'][0])
        assert (approx_anchor_generator['strides'] ==
                square_anchor_generator['strides'])
        self.approx_anchor_generator = build_prior_generator(
            approx_anchor_generator)
        self.square_anchor_generator = build_prior_generator(
            square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator \
            .num_base_priors[0]

        self.reg_decoded_bbox = reg_decoded_bbox

        # one anchor per location
        self.num_base_priors = self.square_anchor_generator.num_base_priors[0]

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.ga_sampling = train_cfg is not None and hasattr(
            train_cfg