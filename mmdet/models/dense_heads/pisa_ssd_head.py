# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
import torch.nn.functional as F
from ..builder import HEADS
from .anchor_head import AnchorHead
from mmcv.cnn.bricks.transformer import build_transformer_layer






@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 n=1, e=0.5,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 transformer=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.num_layer=n
        self.norm_cfg = norm_cfg
        self.expand=e
        self.transformer=transformer
        # self.m = nn.ModuleList()
        # self.m.append(build_transformer_layer(transformer))
        # self.m = build_transformer_layer(transformer)
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

            

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.c1=self.feat_channels
        self.c2=self.num_base_priors * self.cls_out_channels
        self.c_ = int(self.c1 * self.expand)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # self.m = nn.ModuleList()
        # self.m.append(build_transformer_layer(self.transformer))
        # self.m = build_transformer_layer(self.transformer)
        self.m = TransformerBlock(self.c1, self.c1, 2, self.num_layer)
        
        self.cv1 = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.cls_convs.append(
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls=nn.ModuleList()
        self.retina_cls.append(self.m)
        self.retina_cls.append(self.cv1)
        
        # self.retina_cls.append(nn.Conv2d(
        #     self.feat_channels,
        #     self.num_base_priors * self.cls_out_channels,
        #     3,
        #     padding=1))
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        # print(cls_feat.shape)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for cls_conv in self.retina_cls:
            cls_feat = cls_conv(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        cls_score=cls_feat
        return cls_score, bbox_pred


class TransformerLayer(nn.Module):
    # LayerNorm layers removed for better performance
    # c=256, num_cls=27
    # q=(1024.1.27)  k=(1024.1.256)
    def __init__(self, c, num_heads, num_cls):
        super().__init__()
        self.q = nn.Linear(c, num_cls, bias=False)
        self.k = nn.Linear(c, c, bias=False)