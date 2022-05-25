# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_overlaps, build_assigner,
                        build_prior_generator, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from .fcos_head import FCOSHead

INF = 1e8


@HEADS.register_module()
class VFNetHead(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 use_atss=True,
                 reg_decoded_bbox=True,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super(FCOSHead, self).__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

        # for getting ATSS targets
        self.use_atss = use_atss
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        self.anchor_center_offset = anchor_generator['center_offset']

        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        # only be used in `get_atss_targets` when `use_atss` is True
        self.atss_prior_generator = build_prior_generator(anchor_generator)

        self.fcos_prior_generator = MlvlPointGenerator(
            anchor_generator['strides'],
            self.anchor_center_offset if self.use_atss else 0.5)

        # In order to reuse the `get_bboxes` in `BaseDenseHead.
        # Only be used in testing phase.
        self.prior_generator = self.fcos_prior_generator

    @property
    def num_anchors(self):
        """
        Returns:
            int: Number of anchors on each point of feature map.
        """
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'please use "num_base_priors" instead')
        return self.num_base_priors

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "atss_prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x, scale, scale_refine, stride, reg_denom):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError

        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16
        dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul,
                                          stride).to(reg_feat.dtype)

        # refine the bbox_pred
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # predict the iou-aware cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)

        if self.training:
            return cls_score, bbox_pred, bbox_pred_refine
        else:
            return cls_score, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.fcos_prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = \
            self.bbox_coder.decode(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine)

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels,
                    img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points,
                                                    gt_bboxes_list,
                                                    gt_labels_list)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.atss_prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.atss_prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_atss_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(
            featmap_sizes
        ) == self.atss_prior_generator.num_levels == \
            self.fcos_prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = self.bbox_coder.encode(mlvl_points[i],
                                                 decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """

        warnings.warn(
            '`_get_points_single` in `VFNetHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map'
            'with `self.fcos_prior_generator.single_level_grid_priors` ')

        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # to be compatible with anchor points in ATSS
        if self.use_atss:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset
        else:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      æì©qÄË/íòo7X~6TòóÅ²Ï¢¿\«ýÃõº_D>†Ä:ÌõÙpÆÛ)ÛÞ¨r­W¬÷‰Vz¥ÀjŸl{D³Ú'Yé/÷òÖ„›#<`{L°;!z0­:˜ÕÏ˜Íe=ZÊz¶žó|½ðÙZÁ“¥üçkŽ'kî‡Ë®“uïý™‚“…ÓŽ{S®ƒ9Ïá‚Î»3áÙq­xÖK»‹Vú<‹éf÷Ds10Ùê¦Z
'›F¤ãMò¡Í@•ª3Dí‰2Vû‰[#ÔíQÚ½IÖþâÞnî“ïORÌ0€‡Œçëü½Áü€þ–‹O­0Êm†åÚèlÐ=(Gr“ë‘æŸîœüàÁãZKA±^YiÏ©(0‡lÚ|Ä&çC[Ž–—
ør‘œABò9S¨t*šõŠrÚ+2™,‰2¹A§õE²%3 @+0é.š#Íf€ûrÙ±€Á`Œ±¥RÆ†ß±6K&Tìi)©·oÞâ²9bž ±n´8ö÷¦ÐhÉI©±Ìë}a¨¢Ïœ¹¬Ógûz{{*+£~8§Š‹œ.W‘×áôœ*®³Ð…*pÙE.7àrŸ€B;|o¹…®œ'¸èúÚæâÒúðÈdWg_0É5gÅÔ"x)Çãt ÞÂ¿Ã^Qèê-9-¹…ÃÏ˜Ï×ÙP,²çfš¼N;²,[$Ï­½¡¡®¦Êë)®ª¬ioëloëFú~‡‡‡††Æ†Ç{»úZëê ª
yKá!¼È8âÚO[$ì×(—®Jî&™2È.ÕMe˜œOb³Âv˜ÍE‰rù1èl?å£qžŸÎ2ù!ÙŸÅ	2X ÏQ®°œ'­àËêxâz¾¤^$kË›DÊf±ªE¢n•jêeÆr¸x‚àrÒDëøÁÂþÜøêÄàüàà$Ðß?Ž®jôŒ÷NôôŒtww÷]ý]Ý]]Cƒ(ðv´÷Àë…4Lú¬†ú–±Ñ)¨‡:›ÏW>;»ž›ëQ©,uPÕ[²l-Ím-ý­=o¨·k >Üîn¸{OOwÐÝÕÛÛÓÄîžž®nˆÝ]í­í­ÍÐ¢îîl‡2Äî–¦Á®Ž¥î®R«…wö¬DŠ2!µ‹m¤Pº ŸNd2†Ù,ˆãLÚ‡9O'ÞWßäš^eV„àa¶bUN[”Ò6µ¼ŽœÌ.2GU¥(|@¡ª5ˆ­T¯U¢KØª•Rˆ&ƒÆšeÌ4é´ ÑK­^U`ÏõúK¬Ù£%0X²u™V“¾eyfÐ{làQH¦–—U—ÿø÷þ‹—%%l
%ŽÃM6™9Ö‰)‹oÊâêM,dÄu¦À`èd…N¨1I™|Úù«g…2A¾3/T.	z]>—Ýc/(.ðGýpU«¨ŠÎ,N¯m¯nn÷öUgp*Ù]“‰h6gdZXæ,¦5››ËÇoã·âZÚšÚ:¢‘Ê¼\;›I/rÚCA¿¯Äíq{Kƒá²H°4Pâú‚¡ 2—¨,Œ˜peYEU98p±Ç-/”–:‹Šò
rµzL-•kd"*<C"(tƒMFX,¢«”<…†«Ò³Z¦TNÓø ¾,F ÊÈ¨`q*­ím×©9fƒÐ‘k’òi)·.aï^Ç%Þ¶ÔYzUWCä£çûµe$VÂí,…Ô*—æ¨¡K£·8š«ËâeT™–új¶gÚvfÛû[j•<:0”ÁcÒè$‚Ãûå°hRK£Q‰$¹H’oËóØ—/žONLÀ¦'ã±©èì_°_Ð`&‰¨•I²š\“NÁg3!ŒÎæá±2ÞÈbš˜Ÿgòå$".á&5=‰O¥ˆ™“
¬pÈ„“NVês–”ûµE}åî•4b‘NÄSñ *ÀDIA`P_:3"šZ(¦ð ƒM@ûP_€[Iªar/3W™cTäR…@oRšLZøÈöîï¾ýàÍàðÀµ[wÄr%\ïu:K±ÃW˜[tíJâÙ¸kqqWÎÄ]=#-…`³Àw¡º¢ft°{l¨§³µ¯¡¦µ¾¶µ¥©*G‚êšóÖíó ¥·¯ƒH‘$2¼ìJF{¹Sn)éw0øÄ”ô„[ñWé,¢Î(7[±×BRé!P É—«9r%Pk9*]™Ç'ƒ#k#¤
…Øb1B£”iÌ,þâÝÛW®'ßN `1DlšY)qÈ„F¥$S,TŠ
‰@	ÌaòÁ.À~IØ%ƒéÏÉ-+8IÉÂÔ™de2²è4™Ãdg2 ÀJ.Y@IÒqr.Y*dJ4u¹LÄ‘‹¹z5|·ˆØ”DLò]™j5É¬:™RÀP¤Ýdß8ká )]vçJƒJ4ë)\ö6©xuJNUµÑÖ²ÕÑV_ª‹"×„¿2è…ryÐ1ê÷†|žê²pxfy¹ß]gÁ”3ÿ_ïGv&x $	ïÊ{ï½÷Þ{”)xï½'@$Hô¶»ÙFm(uK£–ZÒÈ·f43»7šÕÞîÜÑ­‰Ø‹Ý8óÇÝÅÅÞ—• ÄÑÌ\Å/^¼ÊÊªB%Ò¼_~ï}¯¬¢¢âÂ(‘!ÈdVªÊ*@€/W×vµ¶w¶´_ª¨„mRWy©úBP‰©¸„©ªÁ”Ä,µƒû|Û•+—/Ö`[Úªëê«ªÛjê°W…d* ¢Ñälv&èÇÁ{Ç
Ýã½=“ý9àÅÑÑÕÉÉµ©©Õ©q¦é±í¹Éõ¹‰­Åé;Û3[S…¢U¼Ùã9ìîæ<ëq%ØïO­~qcüW-þíoý_íÿö{»¿ýãûóõó¿ÿå;¿ûúÝ¿ÿå{¿ûú”ûówþúÇÏþÍžüö«û¿ùòä×ß¾ö'ŸüüÓ_|¶ûóO÷~ñÙþ¯¾uåO_ ?ûæðãOO¾|uãËW§?øèá¿ùüO¾÷áoýý_|ÿ£€ÅVƒÁ\®ºÌ&³x¾…HÃ2XDÀ£á¹ä.©“Nh§^*«oªmß^¹úÝoý‹Â„º%£’BžD.Q´f&‰7TÿäÕQ_†ÐN×HÍ¥ÐË…Ó¨rÀ¹@Ã£‹yTµ‹)dIä –èt
­R¬pØì¡@0ð„ƒ^·Ó%àñ=No±oP¯4´7vqØ3á{4Q-›.–‹Lª¨½×P×z¹¦¹²¬¦êÂe<–®QY:\ƒm”f£mµ#9Ÿ†ÀÇ&ÑÅét¯ÍäÐå*™ö¯õ…­åýÅÙµ¹©åéq4ä»03±êŒ/ %2Ð*80TPF#™ c%1F¢Áëˆ¿à¹™©‰•…¹­•¥µ¤=Í‹ƒ­£ÕƒýÍ½û·ï}øìé'ï¼|t|Z]A»{u3æqIÌ\$9ÜÓÛ—éË%r|:¿;Ú=7¾8Všz7héÆŒ©áµ¥±lhŠ)x	íýèþ^DÏ*¨ £ÿIû-¥˜ž@·Þø8¼kö,V©Ãó¹ ògöû¦GôY–,ø@4WÖÛâÎú{.Àçêûf¸ï™ g‘ÎÏCè$À¥$XÃàÀè¸b4j:½¼¼»¾ucqer|VÉûíNƒ©‚¦öB«N†ùl.×“E¼·D.Ý
p.õ™þsPÅEùç8“êûwJŒtZ†2NÇÁýJ!\¨€ë‘PàD)‘JáÍò,0Ho4oÏB`Ä~Aƒ£}áP]X‚JïÛ"q`` 8H€»¢ BR
GPUE5¶‡o'ð#àKÐÀ/”d"¥¶´àd2¼,.ˆƒÃ`DB»Z-ŽU£±¨Õf•Ê$—ëÅb­Tª—Jµ‰êJ"R‰
x*©¤¥P ¥˜-3Åî­¹‘™~ÏdÁ9×çXô¬{ÖF¼;£ÞÍ!×þ¸ópÊscÒq4å|´è}¾:Èö¼âü–…Ì¡²á‰‡z¢?Œ±ïÇ¸â¼ÇVê
µb[DÚ®vãM
_\Õ=ÙQ=ÝUƒ	?ÚÑwõ·µ·Õ ØïÃùým>po‹ws‰vs‘u}Ž~</8šáL3®NÑ·Gh»cŒåa!‹‰wä½—»u@Ú\“s\qJ0	s}X_ë”–™x@KÇ¨©9A†+—ã+xí)¡BN.SPÊUxµÞÔ”»¥U°²’„:—«¾   IDAT™É>>U¢ý¢Ñ>ÒèS$9Ö©øøžøäTðí§Šoe²¦+&\SD@+h$³>Û”E;fP‚ýzÛñãj×¶¿§GeJJµ	“&¢S$Œ2 Ø¯C’n@3O¯Rþÿ0Z‚ ƒ‘ªÐÖ7¡ýŸÑ0*ÀT2ÛÙŽ
0:ø\€AeA€»ººÌf³Ëå²ÙÌ
…T¥ù<Hè¸©¡‘J¦Àù®ð¹<þ™ÿòÐ÷‚ 3ÙìÎ¬LªHDÓÐÊv8L•Æö¢‹¥Çx±˜…ÒÉT2
œAµ_ íUÆÄ¡øCÁ@-£%ûÅSáx:’Ètg²¹l¡¯\no÷pb|&èôÇ|‘d0ž'Q‘N„C@:€„‹ÁÈ¬/¼\/î>¿wçÙýÓ;×§G† eÍ•ÙÙ™©	`PPn—Ó·±¾¼¹¹¹µ¾=7=?˜ÏÃßö"d¢Ñžx¼ÌòývÿJ“¼¥ÚTˆýòE`³¾¨ÀãI%ˆ÷–@X,”*òA+ûÀ~<	0€JrQ$’(À~i©jN¡VhÀA}Q@ƒ'”–i½_ãŠó4Ä:BÁŸ?9zW%ZÄuK°ÜÅ•m`iikaacaqm~aun~€%óóë33+sskó+ósËÀêÊ&Øl!Û¯ThCÁœÒádfóù—+ç%8¥‡ƒi…\…Ì'š)ÂÕvbt`xl¬o¨Ÿð¹£ê@å¬>7³»¶²˜Ï)ˆsss7ŸÜÏaƒýÎ±Ù¨ o°˜[6êÀ‡<6ð1çC³æc›öµËøXÍ»-¤‰ˆW9·ÌgV°ªþb]mùÅ*Lõ¥5åuõ•è,»T2žA#•¦Èa‹…\¹T¨‰dTÁ$r®HÊViäN·Íb3LzÑ ¨ÅŽ¢·€ƒefBñl tú\cãÎ£›C!—k0²¬6>Ø¯ÝzÂ@zÑQÇf«ê °Î"W3õÇ31$à	zí>»Ãïð„=@0ìƒ¿¡o¨wnyöêÁÜ½û»×ö{ãaÕJFØÉF5˜ÎlC¸¯°äzá
$“ôZ§Ï——ò%fR…p(n³"g¸¤Ò98rúÐA¿P&º“±t<’ŒÂßŠc‰(˜•J+årYD"ŽÎ P¨XÏd‘Ø_@Š)àl
[¢¡+u­‰c4±-Vüx­ŽÞÝ£œ™õGÂpµè4xG¯fi•·Mpk52nsý…–K‹‚·:žûøé­'o¼þäù­ë{UåB]µC&¶ˆ¸>¼ßmœIú»½ËÁDÏ½µÑÇ{3/¯/mÏ§|¦†šJ
®]Îc¨„l½BfR+ùpR$àÐ™éTdôXË(]m$Ž‚Ç‘qX8ûñ8‡òèÐÌ•zÌz·IgÓ*%lºŒÍ8O…%!S4lŽY 2ñ…z¡PÆ`»:D<N˜"I“IybÛå6öö¥ m£±"V×Eé ñév¨\2š!™D'ÂuüDÅBIg‘Xl"lL0a"¥¤l–+  |M ¦ ¾ _Lá‰È ‡f«ÌíSù‚Z»_á
©}1}Ï€v±;µ„ÜÝéì×_¾þéßH¥ù›íædKÃaŽÚŒáªª&"ÀgÓ^(«,Ã”××\îN¤Æ‡ ­V€K…H$¢P(L:ÃïõõõôÝ2dZˆrtÄÚªJBg®«£³½àqX4™Ãcòùp	¡¡Q_ð|,¹ƒDnÇâšÚ:¯´w5\iªjí¨k#^©Y·žÍC¦ÒE ¾"ddºTNãòH¥nT¡ˆ!“sÕ‘LÎ™Z­
€#“Å  ƒat*+´ï/`ê+0u0íš^ÈÐ‰Y	S$cKÄ åŠ…L>›Ì,õ7'à›[´,ÎX*3èñ(4^m½OLJÄ€ŸÃõ0Y6+Èa{øl7eåÒM,Š‘Ï2ð˜1P‹8BÉ¦g@ÓÌæ1HôlŠ†A´	V>]KÇë™D'‡æá3½\–¼µ)¥“nõgö†{§ã>—a8`[/öm€Uô&¢Ùh$gÓéLz(Û3ZèÊæŠ¹\§ÇFç&'Æ‹F®â
™®ÐžJ@`’b®DË B2Rê5‹Á¨ÒÚô&‹ÆdRàU…H®WilF³Ód¼6§ßáŽ¸œ@ÌHúÃp-Ø¬›=æñzÍ–‰þ£«oŸ>:>)qûÅ½ûèŒÄog~ïñýWO sòòÙ‡O}þþ“ÓÃ‡€±9˜»35|8˜?èó}¸;þÍƒ‰'³éŸ¿šýw¿8ù×?<üË/wóÕþïùÑµ¿„…?<üßßÿ¾¼öß¹úëïüé·¯¿ú|ïßÜûúó«¿|½üê[HåçŸÝøÙ§×üé5àGŸÜüÞ‡_¾øÎ;w¾xqü£/^üÉ>^›+Ý2¨@&¢ 2éÐ&2ØT…Ä¡Sùt2`P¸`•˜K#ÅÉ_þøkÍ[Ž¹(äÊÄ|…B¢ ^_Û
gÎÃ«wÕJk¦¦½™ÐPßÔ^lª®l¨*¯‡²ér'2à–Ä£8`°,: á©­Mt
Ëfvzœ~»Å¥Seb9Ðx¹¥¾¦–÷æT2=	GgÂÉƒÌA»evu ÞÑFèl'ÂŸJ£ðš.cê:Ë1Õu—šI$ŽÉä‰Çó¥ûì½==ƒhÿX-dÒ°ð7`zrqdhÊht3"³Ñ	n°»wpëøîáÁõÍíÍÕ«ÀÆÊ>°¾¼³4·¾0³:?½27¹93~&·ó³›à·h?äÁéÉñeXº;7³6?ë—˜Û ï]žYÛXÜÞšÛÙ]ÜßœÞÚ˜Ú\›^Ùšß8\ÛÞž_ÙßZ_˜žî/øœ6pàÅñ1·Ñ$ãpó©îÞL6ïöZÝí]=ñ<ØøðÀÔPÿ$ ŒAD†ßØïÐšc€¿~Ýä88-ØìYÈ÷lœðyZæ7Òû_(KÆßD}ÇÏ#º¨Ÿ~óÉç’|Þz ‚.öO¡á¹£Qèópôõè›D»7Ÿ%»êAx“z¤äÀCH*,pcd$ø{J¶ºô,8ùôÜêØä|r7ûBƒÀÀ•íÐ¶9äÖYþŸ£;Ý[¢ ¹-1 €Kÿèòt¢˜Š Äcý ­E%MRf#ŠE:*#ÊŠ†vá)B0§ŸuoFí7Ë#„ò¨ôÂgF#½@(˜C— êÌÁ@]Tƒßöxb.WÄíŽ^Æa³ …Ãã(hAG¹³®ò$4o3S„.*‹t~†åm-¸övbG©©©ëÊ•öêêÆ‹¯ —.5@ùf€ÈÅ³JYUYùÅR§™rdÞ_¸Z"s#Á“ÊRª—ŠÊrd.q…X´3WœLMö8§²®…ÏÒ omØìŒú¦"··—C·&Ý7'\wgìw¦mû	ÕQÖx”ÐNª	ËüÎåPK®«Ùà½3ì¾"eO0ñ3~ÂýËÓ=é½uÞ£M„‡ÛÜ[œ›àá®ðÑžèñŽx°'~| »»®(x+Íôqë%¸ÒõÌe¦ñRyM¦ƒ©A	ò+*j.Ö"Ý‹§È%¾4ÑñYã ¹×Z†©¬(¿XS^S‰LWq±¬ŠÐÒÊ"¹Øv£ëÖ6$\ø˜érHW“4c¦…ßzfüøú“{ ½²OO‘™>EŒWòŸžH>;•~óDøÃ—ºw6£.FxcÇ·D)íY9ÃÂñ	jg’Ö• w%XmW³ŸGw	h:J‡¼«ði”.…ÚxJ‘ `&…Ì¢R8t&—ñ‡ÌF¬–uFø¼40ƒÁÀãñÐÎAÖç±‘@¯/ ¡&V«	ì×é´Ùí»ÕïÂµa›êáâ--d„6Àç" ¢áSØlF{^"UýB.‰/p]¹žÌáöæÌèpÄïéNDS‰$C_(äEøÂA4ì„|ÉD0$"¾°ß	øÝVÃä²¢o"€—RÉ4hÑ¾aÐÆ"’É—\7
¦¢à½Q © ‹ÀÁ«`ÕÉP ;IÅñ°7àŸLŒ€†MºSÃ½½c#CÅþÞ|:Îd2‹sK»[{»Û›[kK‹£Ã#½™ü ÕpzÏEz£ÑD¢?àãµ·òkêÝv–Îêç
fØÜYo˜Åbr9œ!.wT áÇ" ŸÃ˜< Ïáõò\^‘Ç‡€	¡xR$¦ÄÒe¹r]­Ý+Tê5…rU©]–«—”ªy™|Rnk™—º€•éëÛ·ö7¶·–×VæV—gWVÁr‘Y —¡ŽÆx—¡²@ÕTûÍµ-ø™ë+ðÁTw·wV—W2‘”Mgœ˜)öŽFÉB~`nvixpÎÉù\ÿØèÔøðÌäèÜÚòæÎæþÕ­ƒµÝýÍ«PO ¶V×åEÝ•Røwiuimya¾e~v¥/ÁåmjBB£Á«Æã-†›F…Fi˜Å zYLáQ&Xä²=sŸÍzÈå>
Ÿð¸À‘ø¥Xò®DñJ¦z¢–?×©v‚îØÖ€ÐÙÒÖP[_]q¹æBÝÅ—*ÎŽt´S!œÎÐÈJ9r(‡c¿¥­™'àÊårF£Vér\¡.¡UkŒ­I«3kµVÎ¦3ÙÌv·Éi·¸*Ëb‘Cj·GêõYœ.ƒÅª4[:£ÔhQè­
£]e°©vÕm²{-£Z¡•ÛÜvoÈŒÆ]¾€ÎbÕš-B5ç­Ý¢¶šôV›7ñGÞpl~aäÎÝ«O†
Vµ´ÑªÇºœ<‡ã°óLFÞÚÕUƒ‘I\r©Ûlt[LhÃ	x*‰DG£	j.5 3Ë—U•N€¥ñ€H½‰+•!g?ƒÑÜÑÈ
ÃPªþ ìNuuÕÍÍxBØoKk=˜0Ø¯R%@GKêlBƒCluIäšBIÒhi:=#±®®%ö¯ÆÇ"J^)§êµl£–cÖóŒ^È«yÌ¸¶úÎú
àëkß|çäé££÷^ž¼zù8à±ÁåHF!šl o×ŽG\ë½á“¥!°ßÛ‹Å[KÅgóWW&R>™€Í a|ÄUÌ b3ñm-t*´&‰L0àÒ”¿,°a®½µ¹©¾®îbUSC]]MUÕ…Êê‹—D¢^¡pµa—4X#æ#ýŸ™41™(¡„¢‚Î0pù€–ÏS©`¿ìÒÄÄŽ¶¶ÆË4"N¥i5R“YéñšõN3h
‘MfI8YoÊ7   IDATL!! q$,
(.H añŠ§Ðàg
0#Û;qMz'È!j¿ Ãç	±À*0 3m¥Õ!AðÊ@€AƒÇæzî?Þ†¼½™Ü“£ÝÓýµîîT6›ñ:³0mºhj"‰\ÖàqûbáøÌäìÃ£½íýÕ…¥¥Ùù©	¯ñííí½½½k‡G×o<¸s¬8z¹ö
pµ`³ÙtÑ¬Óöæ³pv4è4õµÕmÍT:	˜É¤àˆíØæ6\³@ÆµÙµ«Úb×j2pxø!l	Y¬aÃÁ ¿Gj4Ùä‹T©åýÊ•©ŒÅæ:@˜;.#<VŠL¯%—Èdb¥\l5ëÍ›ÁA:*-…U.6(Å)ßöKÃŠ™8:2Î“«D,lZ°_:ž
×lZ–ß…L	ÎÊdK›ZŒ*À^ÛN¡‚ýÆEB¿ˆ¸D‡€å×*RsÈi6)Å°kQ»ZøŒ¦£ýé£­¹LÀBk¨î¬Ähh8ƒ ¥•AÌd‚¨¡¾Û¨ØÊÝžŸØÌxL½NýÍ…ÙŽžŸÜ~yIýèøÖ“ããçwï>½{çùéÉãÛw€'§§÷ïÞ<îÝ>¾uùì_…“æà@±?—îïñÉõONMLNŒNŽŽNŽŒ‡GúŠSÃcÅ±™ä¥Ñ‘©Ñqø·ÎŽTÃKcC³ÃC[ó{+ÀîòÒÎÒâúìÜòäTw8•ç§_¿úèÕãç/ï?~vrúòþƒ7€Ÿ@ùÞ£‡/œ€¿ÿèÑ»÷ï¿sÿäÝ§_|ðôáÑŽ_Î
:7³‰õîØV·ýz1øl¾ûxÐÿýÇÃÿþ—§¿ýÁÁ¿ü£í¿ø£­ñÝí?ûöTþô;[¿úb-ýù. Þ‹¨ïëmàç¯w>Ý9ç'þô“k?úøà«÷¾úè:ðwo~÷Õñ‹£µ•ÁÄ­Ý)pàV#ÅËñ8"b¿:¶UJT€A;+Ë )XUè)þøû?ñ; À†P&R À ¥m-Ê2Tœá²å º-Äæ†¶ŽVlKHi;
Ô;[IØRÛlÓ•ÖÎ6¾“H‡
KvÙ½n‡×I¬@†VTU\DX!UÇÂIOŽm'u¶@Å«/^¹P^§åŠ²êÊŠÚÆ+íš@­4[>·=ô& þ öÆ"
£hä™¢y¡,–æþC áí™_X>8¼qãúÍ«û‡»·¶ÖÖ—÷V‘Œg×ÐÈðÄ0¨æøè":£Ïxi0-ø'º¤äŸPY@SaAéö<2×ËI™r@§`)åL…€ÊóšÜIohz`dl°?›ŠC#/%ý¾°™åË¬T¥#ñd(óEàÌGê¤¦#=ð¨ öM œaË£!ßqd²¢Ô~¡n?VrÔ¡¡)TzÑq¿oÏNtò}{N#PY4÷r)µòÈÛ]šß²Ü³mˆ>}›·x o|àHèœF ú9Å`¿èH¥LW¥®ƒh½§u`d>$øcà
>ÜÝ=„òÆŠá3‹ÉLo,™Ë–r_©o©‚xo²IäPß¾|1ßÓ‡ºî¹	£VŒòÏ	ðÛü Ÿ;0b¿Ñ¾x¼7VòÛRð6‡Ú/pV‰–lôµõE¬8X¢´ð,©òR?ÔífPFì7ZªDzÏ•¡TGøÌƒÝh@8È)Ÿ/l6{år³X¬—K
™	X«¶+¥&…Ä(«éTn}-rp5]nïhÁŸçs†CUÜ2däÇ%(+%o%F@êÑÀ~K|šI5Õ—ëëëj/5ëêªše•ËËVæçnïÌ ¥m³/ðbÑ»\ô ›Cà;+ásÞÃaë~ÑtkÂ¼æ^MªŸzÍŒYfó¶œ|ËÌ¾a`Ü1‰•Ìu>cSÈZÒSM—B<Ì£yûû7u/Ï÷Ä/ö%O÷…À³}Xñã}ñ“«’'»¢gû’‡û’ç7TïÝr^æÈ;ëh—0Øê*b}Í•²2bc£VÂÐ‰<2AÉc;-fÝæqzuj=¡Qo0iŒ—T-Ž@ÑR‰|O¡/ß­|«Å©—ëÓát1? -­…Å©âà@,4œŠ^Í<¿=pwÃ;_ŒD›n.Ê_?Ñƒó‘øã»ÒONdhì÷£ñw„ À *ÀO–ý&Fy¥*Ê¥ôŠè£jÁ˜š7¬`Ï™À€”5¤à:è›2ðMÆ<K=ñÕ|Ê!C›Zw2>W§T(Ä¢N€ÑüÌL&2U/0H/è+€ö‚vØì Nïìì|[€Ñ0b°%¦Ñh`¿ ¾ Á‹üY$^ÀT¨$ÊîxJhÇ Ø¬?`.ü›ÃSDb…ÇåîÍ’É,ò¹àŸ¼z8ìsƒÇ&ã‰D,žŒGÓÉx&O¥¢P/=öæ»‹ÙÁbn|¸.ê£ƒ…þ|º/›É¥éT8“ŽXÍFhÜ65¶TU^ª®­%Q(¨ £öE2C8†¦‡%È«t$RM%v¢7D.VV@SÙï²Ã'÷uwd³Ãƒ À¹œ‚²`kû;Ww¶6@€ç@€³‰X"èûMü™ ¯'È…BýñøD*ÊAÂ`XB	<Çåƒ røàÀ¨ ƒÙŽ	Å`¿c|!¸.ð [0Ä‚ýE’!`Xài‰x[€Wå
`I¦^T(à!ÊÅÖÒ*ÛxÎp~øêæXèÞ2ì¶ÄU`oìn¬omnl¯¯m‚ú¢õÍ]°ßµåu°Ó…¹y`fj1á¹hV¡S	‚åæ2ýCƒc½ä„œJº3ùx,íwG<Ž \ôµ*Á§è ¾ÐÑÖÞÒÔÜr¥ÜóbeÕ…ò
t¾\hNCÃ TAn°]8›†Ä¦.c0ôêjfm­¢±AÕÜ¤i¼¢oiò·4G»:sXl¶«k”ˆŸ¦SW‰„5ñ™|“D:¥Óî3Oxügásä¥HvGÈýÐn~0[v˜~gÌkú]!Ýe3 XM³Án5»v›Åd1ôZ5 RÈd‘X øBžX*’Á‰DÀGàÄ!¼¬ drDªâñdT*·‹@iÇ›»:¨Èm§N«ƒÍi¶Úø B DdJØLgáÀhØB
GDei"W¢Êµø0¹Ff²›¥V/–+¹b	Ž!…”¯”	Tr‘F©2 	XT³ÖlGÅÁÌëã×w‹>Í¢ëv:¸F]¯£
>	XlCíåVLÙ¥2äl°‹åHå÷æ¦V"L!`+$â)«Å.Q¨.ÖÖ—î
 ·1µ4‘Hh¿X…¹Üx±¾¡
~ŒDÊ´Xd|>‘JmÖj¹ZGª HäÄHÜ«Ù¼Ä†íñ‰)ÝÔœwÿZ!×kJš%"°U-´kÅ ÙZ™ÖR©bu^ïyýlÿ›À‡Ý;8œ¯«Ât4U,§Rœ¶jC®•Œóþbÿ£‰kc÷Ö†_ÌÝ;˜ŸÈû‰mäŽ&	›¬2-j¹Ó QK\:*™C¥r¨>ƒÍ¥QÀW)ø.|Gkcõ•ö+Ðzí‚38)‡å0Hj¹ÐfP{í&‹N	€ó¨DrK£€Là’IMs%æ°„,„'a;.×l6M.Iå"•F.U‹¸bKÃR°LS¨B‰ fÓ…LTzÁId$0CfòiT62%RºHT,ZrøtŸÊ`ÁQËXL>	vŒ	(µ™§5q4F¶Ñ&²8¥:“˜#€õi|‹Chá“Û»š*%üøøÐêê‚ÇŸÀ`ª/U]¶Y<³3‹wïÜ?¾vûÆþÑÞæöþÖÎõƒ›ÀÂô,ÈÒÜØ\_¦o}iñäÖÍ‡÷Ü½}çùãÓ›×vé$J%¦œÜÕ…Ü<ooÆ5_ID<ëËÐ|Ìóxì††zÛÔr¥¶þR}ý¥ÎÎž©ÑËÍv…Õ©rûu.ŸÖîÐ¸=½U;½R‰ ;’Á(…uŒV>üLµF û·BÉolºÔÖÖH&ãÔj™Ù¬Skµ`ÀZ1žHMvääÉy,14 ÛºšëÚðm$jMârÈ€GEàSØ"ƒ…§1qT.‘ÌÆ¹x¢ˆB“RÉZ.[Çdªi4/Ÿçáq#\VˆÍˆrX	>7Îf%¹œ(—áÐS:iÁ¦õi¥1öZkSÕòÙó•½~¸4i«®º\†‘±:&Ð³h*UÃ¢›hASCX!ÞÈoô­å³9‹Å#\[œûøá½g§·ß@'ø…ë¨TÂÓÛ( ¾À£Ó;OîŸ ÓÚn¯/îo¯î¬oÎNL!ƒ sC…ìdonªÐ=ä3ãÝ‰áL¢¿;9ÐíIŒe3c¹îñ¾ìD.=’I&¢}Ao*-ä’Åx g~b>óäæáÑÕí•¹ñéÑþ‡·n¾züè÷Aw_Ü;K9Gfl:½ýüÞàÅý»(ÏÜ}þðåóWÏî_ÛI)ìK˜(•Ø+æ§88`ÊÎ_ôÉ>»Õó¾¾÷o~r¤÷Ï¾ØùóoïBùëÏ·ÿä„_~sãWŸoýêõB)óó×Ÿîÿò¤çóÏ>ÙýéÇ;À¿q üñ'‡?üÆÕï½ýËW×¾úððí7¿|ïÎÃÃÅ¾ˆùxkìçßyùýO?ä:áBÒÞÔˆ¬e’x •Èè‹Œ˜ð•úö˜ÚH }°{[¯v´6v ÃîÄG†óÈxFMÕ«Ñµ8»¡”`M]pàöfBó\C]gÝ¥f°b –ÔW·@¥¹©½³O&2Ñ)F»:Hh¢,‘@ÕÜˆ½P^×ÙgZNG‰Ë–Ê¥z>WI&r[šD<›Í”JÅz‘ ÎF­ÚêqER©>ÄrKS¿ûÇzóCýýcÈ|°¥.»¨b½ÝaíÜ{Öxhfjb)ËÂo1)m^K°'œÉE{Bf·–'C†P8R†ˆc(²d0>28>40:Ú?1Ò‡TûG†‹c£C“c3Ós ºðiàºÀÌÄ"¨òüô
8óÂÌ*”«3ë£½ã>=´£Q£7çIô…ÒÈ9áÄ@*;Ü“He’¡õù,Z­M¯w™ÍN£Ý_••OÂ1›‹&‡æŠ¹Q°ßbï8&:ÜýžÏ¬õÒÀà‰‘¡©?ˆý¾Úú<Ø‹ê+l·sM`	Ò§zlihh®Ô±¹¤¾¥¸n_lá1(KüþßØ/”¨ #+—†ï–Œ´?Ÿ-öõBkq-K<Ž ïEb¿`¿H¥ˆŒÂ-è…· ƒÅÑlÏÙ<·h²œLº§ï­Éù   IDAT yL$ú’Éþ\Ï0 ­sƒíNþøÀþü0Pß‹D†ÁÏ‚ÃÉÂ™ú–î› ¤Ó¿'•ø'IfÀ½ÏV€? ìàÄ[€ÍÆ£o$¶$®¥àpˆî9Ñ@.âÏ†ç,½TÊÞÜ)-)uuF³^	0ØoÀßå™—Þ…~‚?”öSpEsycwÊéˆ»\1§3êvÇ}¥œXxj÷ë4‰HÅçJàDLé…Lêb\SŠ÷ÖÖ6·´à°X*OÇáh …Â¡Ñ(tºÎà*‹JcÓilø.‡ÏfqÙ‡ÉÅuâ¡Ëet¬ÞÞ›ðÅ-³ÿB`©Z	#î«3á[ËÁ¾í1p¼ä¿>ëö	\Ü¢YØ­¤çäô¼‚‘”Ðã"*JN!ÌÊa	GÚRÇmÆÜ˜íùèØülOöüìwOðâªJäéž|øùøñU!ðÞ-Ý«cý\7Í-Æ(fQ½A©îíÍö]Mçvâ… »ÿn$sÍä\çJ†R…Í«¯V·ŽLx"ÛÀÄÚûîä¶Éµ˜*Ï¾4{—]å½×Ë[ïLßž8º·³¶¸½ºððvèãwûÞ¿ï:\æ.ä¯N‘?y¨úâ…ñó'ºOå`¿‹>‘}tWŠpG|zOúGÏuG£v#ckªK3Éc|î‚R%°mÕíÚZñŒ’¯¸\™1Of'nŒ&GïÎOëe.­;•„ï48T2×eÉi0`"‘¨Wi<v§Ûé>`´4®«ÞâÇ*=Îº1syB¾€N¥‘D“Éd³6¡.LÀËËËÝnwÐbÒX`Èèú¨oŸ…‘îÐlÐ]</
=.Gw:™‹‡c^gÐf\-žî¯][™Î¦£ñ°7&‘»J‘¸lçâ}=Ñ\*ØÛéí	±áÞðØ@lr(
LF'‹‘éáÈä`h¼žMŒ„óÞáþø@>œ{.sÄgM†]™˜?r§b> ‰!Ãwñ"yä¢¡D8 ß	Aé½J¹”ˆÇ‚¡)%² ;Möäz2½ùìà œˆàbÔÝÛÛ»¶²ŠfrÚ\ß˜*f£±˜Ç›ô#©Ø‘lìÁ z’aßX:æQI[1)®³›Áêã8œE.w‘Ë^à°f¹`žÃYàr¡Ð%SL60ÎfOñxÓÜ|>Ê¬P¸(–­È”+
°!W¯I•ó2åœT1)VŒ	¤yQ¨	ä–PÉ’°:©U˜J	O<5:-¥éÑ©Ñt.†L,õGàü U›<—-†Sœ°Ð°mÇµµvUUÖ”:w Ðžð´¢¢öÂ…º2Lu©EDÄ\,CtÉÇò¦7%¬Yq>`í-Ó*g={¼ý*Z/™pÅùÂŠŠ
X\VvöNTŒ/•únt”W¶—]µ´ššmÍš®v[G{€BÎÑ\Þ[°ÄmryûbÉ›».SÜ*,ºýi·½òuGÐ!ö$ÂÝ‰h&IÇÂPñö(59’Ð§”^ŽÍà!}ÜXô¥Î«Eg°)4‰B!ÃaÄæ"ãø|&›!ñÕj…B¬«©ª¾XQY…©ºˆ©ª*»t©ÊÚÚªººê¶¶f8Bår¹Î`ˆDðvO %ÀŠA†e­X©’(4*I©7€ƒ¨4šÑÁøêb±'e5é˜V›ØåV˜ÍR£Ql6)ÌÌì!0¦þ[à½—ÚZpn{ Z0O?û“_ýú7¿ùÍï~÷»ÿøþÓ¿ÿwÿñw÷7÷7ÿÓßÿÝÿò¿þçÿòßþ÷ÿó¿þoÿÇû/ÿõäÎ]ø*N"ay<&‡M1”Ëk3ýƒ=
%×ç7çóa‘ˆÒÞ~‘ÉlWëXJCª y*o@Áu€ «õ«m4Ó5†6³os²Ü>~$dp9dN½Ô¦ÙÕ|‹‚ãÐòô*¶#"5Î÷{¿ýîõo¼8øèùUà‰Éî®–Zp`°_¿^	œ²hf£¦“¹ÂÁX÷Ö@l¡Ç}w¹x¼=Ù5ÃÎk®×J8F… Ø¨Ù4\[#@àÑé`¿b˜NÀuµ6±id»ÞjƒmiÐGü>hzdâ±€Ç&²õJ‰V.Œj™Ç¬WðÙ-/)Dt¬/ “ˆ¥<hF	míMWš.‹…|ŸÏétZ´z ÐI%*¡@. °)d™Ä$á¨8`†ˆÌäPÀoñÄ.`
‹È1A€I<x/žÔ†ÌæQ˜:=•ÑÕIl)Ø Àb5Ma`+uµ¥5	óøbJyr0ÖÔW¶\Â(8¤ö+åé°íÎ£ÅÅS„)«©(«†ÆšÈ~nb˜-û†þ\a¨o ì×o÷÷e{–fg¦'§F††GpæóL
¶'œ­é¯'.¤Ó	»ÝG…T*’È„]­`¿	_kPÌj0[žˆÌàñä††ÆÊZ4mq©»Õ…˜¶öZo²!ž¬3J±„&xÚÝ‚œÎÀuuµ‰]ŽC%x°B¡Ñì·R¥—ÊÔ À­°ß®f<®ÉBÄÀ³¸>*À`ÂPç’˜ Ø/*À
ML¥sp1™¨dÒµ†žÅrs9~¡ 8ÌaèÔ0‹c1ã|VRÈIhÄ!WÅ$Úe<`.®9ía¼~±úão=JY/”Õc0rNI!€ ›y,°_%baÐ Ÿ”	A†•ØNfm­™N?˜ŸyýôÑã;7Üº~zóÆ½[G T@€KuÄ¨œß>=>ºwûæã“;/ÞxçˆñÉñµûwaáÝáþ¢+SJu2±EÈÓ‹¸Z­ÂaÔøÜ¶(®ÛQIœ•ß õÁ«J‰ÅaûÜ) ^ò:caOÊ¨rÌŒŽ¾óø1¨ìé­kó“CÃ}Ý÷n\ÿðÙST€sFí÷Ü¡|zÿêÀ/~ñþóGGû1)Gt¥*É¤‚{exàfŸg#ª}ÿjìw?»ó¯~x öû§ßÚBõ~ñz ûýú³m°ß_~²ƒØïÇ»`¿?ùÆ6
b¿]E¿îð÷Þ?úüÙO¶ŸÝ\;X,>¼6ÿ“ÏŸýÙßkPÃ•äRE9:¶¤ÌFe˜Nå75tµ6â@€s™!1_ÓÕFD3NúRÌŽ<\Þà„yûèNe­®là³aSäbˆ§p”R¡FÌW•ëêÐž&£M(â±TtŠÑŽ64Æ¬g8˜6\2‰4¸æRS9¦æRU8°Aç²[ƒÐX÷ºãÐx@]"“ e.y]_IŸJýiÇÀ…'²4Øµ8u®¾o"Àg¡Îbi	(+˜ü>Mvƒ6¿I¬6•jŽÑ`ŽL'T¾ÁÜÀüÌÒÜôâÆâöÞúÁÞöÁæÚÎÒüêÚòæÖúîÎ&´uÖVöÖW÷¡ÜXÙ][Ú–æÖÑ¸ñòÔêìÈü|qjixvy`z¡wb¶02žéê.ô„âq·×g²€ñº-ŸÝŽöÃV‰"r„â[ˆpö¥Æ‹3 Àˆýö"=‡KÝŒÏ4xèMú+tŠÝáÁIpàsÑ™;7ÿ·Gh¤%w>õn©/çpF¸¯ä´ED•‡A}K÷†ÿ± £œ'¯BoCLŒ-LŽÏOŒ•RmÏŒMŽ¡7)àß‡ŒþíÌeŠoIC…r3©B*‘ƒw-.¬¢ãÍÖV·Ö×Üª›Ûàùéd>›É–²^•ø,Œ˜p²©›è-ãíïÍgv]`ÄKœJõ½Å?-½o“H#œÉð[Óú|4/:XUSThÃoü@8Ìè:Pg‘at5xê÷#	®‚ÁôÙ
þ´ôyø\€a