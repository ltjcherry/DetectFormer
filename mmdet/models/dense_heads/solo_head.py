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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ��q��/��o7X~6T��Ų���\�����_�D>��:���p��)�ިr�W���Vz��j�l{D��'Y�/�����#<`{L�;!z0�:��Ϙ�e=Z�z���|���Z�����k�'k�ˮ�u������ӎ{S��9��λ3��q�x�K��V�<����f�Ds10���Z
'�F��M��@��3D�2V��[#��QڽI����n�OR�0�������������O�0�m����l�=(Gr��������ZKA�^Yiϩ(0�l�|��&�C[����
�r���AB�9S�t*���r�+2�,�2�A��E��%3 @+0�.�#�f��rٱ���`���RƆ߱6K&T�i)��o��9b� �n�8����h�I����}a��Ϝ���g�z{{*+�~8����.W�����*��Ѕ*p��E.7�r��B;|o�����'����������dWg_0�5g��"x)��t ����^Q��-9-���Ϙ���P,��f��N;�,[$ϭ�������)���io�lo�F�~�����Ɔ�{��Z�꠪
yK�!��8���O[$��(��J�&�2�.�Me��Ob��v��E�r�1�l?��q���2�!ٟ�	2X �Q���'����x�z��^$k˛D�f��E�n�j�e�r��x��r�D�������������$��?��j���N��tww��]�]�]]C���(�v����4L�������)��:��W>;����Q�,uP�[�l-�m-���=o��k >��n�{OOw�����ӏ��n�ݝ]����Т��l�2������R��w���D�2!��m�P���Nd2��,��L��9O'�W��^eV��a�bUN[��6�����.2GU�(|@��5��T�U�Kت�R�&�ƚe�4���K�^U`���K���%0X�u�V��eyfЏ{l�QH���U�������%%l
%��M6�9��)�o���M,d�u��`�d�N�1I�|���g�2A�3/T.	z]>��c/(.�G�pU����,N�m�nn���Ugp*�]��h6gdZX�,�5�����o���Zښ�:��ʼ\;�I/r�CA����q{K��H�4P���� 2��,��peYEU98p��-/��:���
r�z�L-�kd"*<C"(t�MFX,���<���ҳZ�TN����,F ʐȨ`q*��mש9f�Бk��i)�.a�^�%���YzUWC����e$V��,��*���K��8����eT���j�g�vf��[j�<:0���c��$�����hR�K�Q�$�H�o��ؗ/�ONL��'㱩��_�_�`&���I��\�N�g3!�����2��b���g��$".�&5=�O����
��pȄ�NV�s����E}��4b�N�S� *�DIA`P_:3"�Z(�� �M@��P_�[I�ar/3W��cT�R�@oR�LZ������������[w�r%\�u:K��W�[t�J�ٸkqqW��]=#-�`��w���ft�{l�������������*G������� ����H�$2���JF{�Sn)�w0�Ĕ�[�W�,��(7[��BR�!P ɗ�9r%Pk9*]��'�#k#�
��b1B��i�,����W�'�N `1Dl�Y)qȄF�$S,T�
�@	�a��.�~I�%����-�+8I����de2��4���dg2 �J.Y@I�qr.Y*dJ4u�Lđ��z5|��ؔDL�]��j5ɬ:�R�P��d�8k� )]v�J�J4�)\�6�xuJN�U��ֲ��V_��"����2�ry�1���|��pxfy��]g��3�_�Gv&x $	��{���{��)x�'@$H����Fm(uK��Z�ȷf43�7����܍ѭ��؋ݍ8�����ޗ� ���\�/^��ʪB%Ҽ_~�}������(�!�dV��*@�/W�v��w���_���mRWy��BP���������,���|ە+�/�`[���ꫪ�j�W�d* ���lv&���{�
��=��9������ɵ��թq�����������;�3[S��U���9���<�q%��O�~qc�W-��o��_���{����������;���ݿ��{������w�����͏����������߾�'���ӝ_|���O�~�����u�O_ ?����OO�|u��W�?��᏿��O���o��_|����V��\���&�x��H�2XD�����.��Nh�^*�o�m�^���o����%��B�D.Q�f&�7T���Q_��N�H�����Өr��@ã�yT��)dI����t
�R�p��@0���^��%��=No�oP�4�7vq�3�{4Q-�.��L����P�z��������e<��QY:\�m�f�m�#9����&���t�����*����������ٵ����q4�03���/ %2�*80TPF#� �c%1F����๙������������=͋����Ճ�ͽ���}���'�|t|Z]A�{u3�qI�\$9��ۗ��%r|:�;�=7�8V�z�7h�ƌ��ᵥ�lh�)x	����^D�*� ��I�-���@���8�k�,V��� �g���G�Y�,�@4W�����{.����f�� g���C�$��$X����b4j:�����ucqer|V���N�����B�N��l.דE��D.ݍ
p.���sP�E��8���wJ�tZ�2N���J!\����P�D)�J���,0Ho4o�B`�~A��}�P]X�J��"q`` 8H����BR
GPUE5��o'��#�K��/�d"�����d2�,.���`DB�Z-�U����f��$���b�T��J����J"R�
x*���P ��-�3���~�d�9��X��{�F�;���!����p�sc�q4�|��}�:�������̡�ቇz�?���Ǹ��V�
�b[D��v�M
_\�=�Q=�U�	?���w����� �����m>po�ws�vs�u}�~</8��L3�NѷGh�c��a!��w佗��u@�\�s\qJ0	s}X_딖�x@KǨ�9A�+��+x�)�BN.SP�Ux��Ԕ��U����:���   IDAT�ɐ>>U����>��S$9֩�����T��oe��+&\SD@+h$�>۔E;fP��z���j׶��GeJJ�	�&�S$�2 دC�n@3O�R��0Z� �����7����0*�T2�َ
0:�\�AeA�����f�����
�T���<H踩��J�����<������� 3����L�HD���v8L�������x�����T2
�A�_ �U��ġ�C�@-�%���S�x:��tg��l��\no�pb|&���|�d0�'Q�N�C@:����Ȭ/�\/�>�w����;��G��e͏��ٙ�	`PPn�ӷ��������=7=?�����"d�ўx�����v�J����T���E`�����I%���@X,�*�A�+��~<	0�JrQ$�(�~�i�jN��Vh��A}Q@�'��i��_��4�:B��?9zW%Z�uK��ŕm`iikaacaqm~aun~�%���33+ssk�+�s����&�l!ۯThC����d�f���+�%8���i�\��'�)��vbt`xl�o�����@�>7������)�sss7���a��α٨ o��[6���<6�1�C��c�����Xͻ-���W9���gV���b]m��*L��5�u���,�T2�A#���a��\�T���d�T�$r�H�Vi�N��b3Lz�� ���Ŏ����efB�l t�\c�Σ�C!�k0��6>د�z�@z�Q�f�� ��"W3��31$�	z�>�����=@0샿�o�wny���ܽ����{�a��JF��F5��lC����z�
$��Z�ϗ��%fR�p(n�"g���98r��A�P&���t<�����c�(��J+�rYD"�� P�X�d��_@�)�l
[��+u��c4�-V�x���ݣ���G�p��4xG�fi��Mpk52ns���K���:����'o������{U�B]�C&���>���m�I�����DϽ���{3/�/m��|���J
�]�c��l�BfR+�pR$�Й��Td�X�(]m$��ǑqX8���8����̕z�z�Ig�*%l���8O�%!S4l�Y 2�z�P�`��:D<�N�"�I�Iyb��6����m��"V�E���v�\2�!�D'�u��D�BIg�Xl"lL0a"���l�+� |M ��� _L�� �f���S��Z�_�
�}1}πv�;��������_����H������dK�a�ڌ᪪&"�g�^(�,Ô��\�N�Ƈ��V�K�H$�P(L:������݁2dZ�rt�ڪJBg������qX4��c��p	��Q_�|,��Dn���:��w5\i�j�k#^�Y���C��E��"dd�TN��H�nT��!�s��L��Z�
�#�Š��at*+��/`�+0u0��^�ЉY	S$cK� 劅L>��,�7'��[�,�X*3��(4^m�OLJĀ���0Y6+�a{�l7�e��M,���2�1P�8Bɦg�@���1H�l��A�	V>]K��D'���3�\���)��n�g��{��>�a8`[/�m�U�&��h$��g��Lz(�3Z��抹\��F�&'��F���
����J@`�b�DˠB2R�5������&��dR�U�H�WilF��d�6��Ꮈ�@�H��p-ج�=��z͖�������o�>:>)q�Ž���og�~���WO�s��هO}��������9��35|8�?��}�;�̓�'�響��w�8��?<��/w�����ѵ���?<�������߹����鷯��|������|���[H����٧���5�G�����_���;w�xq��/^�ɏ>^��+�2�@&��2��&2�T�ġS�t2`P�`��K#��_��k��[��(���|�B��^_�
g�ëw�Jk�����P��^l��l�*����r'2��ģ8`�,: ᩭMt
�fvz�~�ťSeb9�x�������T2=	Gg�Ƀ�A�evu�����F�l'J��.c�:�1�u��I$�������==�h�X-dҝ��7`zrqdh�ht3"��	n��wp������͍��ի���>����4��0�:?�27�93~&���h?������eX�;7�6?뗘� �]�Y�X�ޚ��]�ߜ�ژ�\�^ٚ�8\�ޞ_��Z_���/��6p���1��$�p���L6��Z��]=�<�����P�$ ��AD������c��~��88-��Y��l��yZ�7���_(K���D}��#����~���|�z��.�O���Q��p���D�7�%��Ax�z���CH*,pcd$�{J����,8������|r7�B�������9��Y���;�[� �-1 �K���t�����c� �E%MRf#�E:*#ʊ�v�)B0��uoF�7�#����gF#�@(�C� ���@]T�ߐ�xb.W���^��a�����(hAG����$4o3S�.*�t~��m-��vbG����ʕ���Ƌ� �.5@�f��ųJYUY��R��rd�_�Z"s#���R����rd.q�X�3W�LM�8�����Ҡom����"���C�&�7'\wg�w�m�	�Q�x��N�	����PK����3���"eO0�3~����=�uޣM����[�����ў��x�'~| ���(x+��q�%����e��RyM���A	�+*j.�"݋���%�4��Y� ��Z���(�XS^S�LWq������"��v����6$\���rHW�4c���zf�����{ ��OO���>E�W���H>;�~�D�×�w6�.FxcǷD)�Y9���	jg�֕�w%XmW��Gw	h:J����i�.��xJ� `&�̢R8t&���F��uF��40�������A�籑@��/ �&V�	��������µa����-�-d�6��"����S�lF{^"U��B.�/p]��������p���NDS�$C_(�E��A4쏄|�D0$"���	��V����o"��R�4hѾa��"���\7
���Q ������`��P�;I��7��L����M��Sý�c#C���|:��d2�sK�[{�ۛ[kK���#��� �pz�Ez�сD�?�㵷�k��v����
f��Yo��br9�!.wT ��"����< ����\^�Ǉ�	�xR$���e�r]�ݐ+T�5�rU�]�����y�|Rnk�������۷�7����V�V�gWV�r�Y ����x���@�T�͵-���+��Tw�wV�W2��Mg���)��F�B~`nvixp���\����������������խ�����ͫP�O �Vׁ�EݕR�wiuimya�e~v��/��mjBB�����-��F�Fi�� zYL��Q&X䲁=s��z��>
������X�D�J�z��?שv����ր����P[_]q��B���*Ύt�S!����J9r(�c����'���r�F�V�r�\�.�Uk��I�3k�V�Φ3��v��i���*�b�Cj�G��Y�.�Ū4[:��hQ�
�]e���v��m�{-�Z����vo���]���b՚-B5��ݢ���V�7�G�pl~a��ݫ�O�
V��ѪǺ�<����LF����U��I\r��lt[Lh�	x*�DG�	j.5 3˗U�N���H��+�!g?�����
��P����N�uu���xB�oKk=�0دR%@GK�lB�CluI��BI�hi:=#����%����"J^)��l��c��^ȫy̸����
��k�|��飣�^��z�8���HF!�l o׎G\�ᓥ!��ۋ�[K�g�WW&R>��� a|�U� b3�m-t*�&�L0�Ҕ�,�a��������bUSC]]MUՅ�ꋗD�^�p�a�4X#�#���41�(������0p����S�`����Ď����4"N�i5R�Y���N3h
�MfI8Yo�7   IDATL!�!�q$,
(.�H a������g
0��#��;qMz'�!j� ��	��*0�3m��!A��@�A���z�?������ܓ�������T6��:�0m�hj"�\���q�b�����ã���Յ�����	������k�G�o<�s�8z��
p�`��tѬ���pv4�4���m�T:	�ɤ�����6\�@Ƶٵ��b�j2px�!l	Y�a�� �Gj4���T���ʕ����:@�;.#�<�V�L�%��db�\l5����A:*-�U.6(�)��KÊ�8:2Γ�D,lZ�_:�
�lZ�߅L	���dK�Z�*�^�N����EB���D����*Rs�i6)ŰkQ�Z�����飭�L�Bk���hh8� ��A�d����ۨ��ݞ���xL�N�ͅ������~yI���֓���w�>�{������w�'�������<��>�u��_����@�?������O�NM�LN�N��N���G��S�cű��с���q��Ύ��T�KcC��C[�{+������������Tw8��_������/�?~vr����7��@�ޣ�/�����ѻ��s����_|���ю_�
:7�����V��z1�l��x�����������������������?��T��;[��b-��. ދ���m��w>�9�'���k?�������:�wo~��񋣵��ĭ�)p�V#���8"b�:�UJT�A;+ˠ)XU�)���?�; ��P&R � �m-��2T��� �-�憶�VlKHi;
�;[I�R�lӕ��6��H�
Kvٽn��I�@�VTU\DX!U��IO�m'u�@ū/^�P^�劲�ʊ��+��@�4[�>�=�&�� ��"
�h���y��,����C���_X>8�q��ͫ������֗�V��g�����0����":��xi0-�'���PY@SaA��<2��I�r�@�`)�L�����Iohz`dl�?��C#/%�����ˬT�#�d(�E��Gꤦ#=��� �M���aˣ!�qd���~�n?Vrԡ�)Tz�q�o�Nt�}{N#PY4�r)����]�߲ܳm�>}��x�o|��H�F �9�`��H�LW�����h��u`d>$�c�
>��=��Ɗ�3��Lo,�˖r_��o��xo��I�P߾|1�Ӈ��	�V���	���� �;0b�Ѿx�7V��R�6��/pV��l���E�8X���,��R�?��fPF�7Z�Dzϕ�TG�́��h@8�)�/l6{�r�X��K�
�	X��+�&��(��Tn}-rp5]n�h���s�CU�2d��%(+�%o%F@���~K|�I5՗���j/5�ꪚe���V��n�́ ��m�/�bѻ\� �C�;+�s��a�~�tk���^M��z͌Yf�|�̾a`�1���u>cS�Z�SM�B<̣y��7u/���/�%O����}X��}񓫒'��g������7T��r^���;�h�0��*b}͕�2bc�V�Љ<2A�c;-f���qzuj=�Qo0i��T-�@�R�|O�/��|�ũ����t1? -���ũ��@,4���^�<�=pw�;_�D�n.�_?у����ONdh����w� � *�O��&Fy�*ʥ��j���7�`ϙ���5��:��2��M�<K=��|�!C�Zw2>W�T(ĢN����L&2U/0H/�+���v�� �N���|[��0b�%��h`�������Y$^�T�$��xJhǁ ج?`.����SDb�������,�����z8�s��&�D,��G��x&O��P/=��滋��bn|�.ꣃ��|�/�ɥ�T8��X�Fh�65�TU^���%Q(� ��E�2C8���%ȍ�t$RM%v�7D.VV@S���'�uwd�Ã �����`k�;Ww�6@��@���X"��M����'ȅB���D*�A�`XB�	<�� �r���� �َ	�`�c|!�.� [0���E�!�`X�i�x[�W�
`I�^T(�!�����*�x�p~���X��2��U`o�n�omnl��m������]�ߵ�u�Ӆ�y`fj1�hV�S	���2�C�c�䄜J�3�x,�wG<� \��*���� ���������r���beՅ�
t�\hNCàTAn�]8��Ħ.c0��jfm���A�ܤi��oi�4G�:sXl��k����SW��5��|�D:���3Ox�g�s��HvG���n~0�[v�~g�k��]!��e3 XM��n5��v��d1�Z5�R�d�X �B�X*���D�G��!�� dr�D���dT*��@i���:��m�N���i����B DdJ��Lg��h�B
GDei"W�ʵ�0�Ff���V/�+�b	�!����	Tr�F�2 	XT��lG��̝���w�>͢�v:�F]��
>�	XlC��VL٥2�l����H����V"�L!`+$�)��.Q�.�֗�
��1�4�Hh�X���x���
~�DʴXd|>�Jm�j�ZG��H��H��ټĆ��)�Ԝw�Z!�kJ�%"�U-�k�� �Z��R�bu^��y�l������;8����t4U,�R��jC�����b����kc�ֆ_��;�����m�&	��2-j�ӠQK\:*�C�r�>�ͥQ�W)�.|Gkc���+�z�38)��0Hj��fP{�&�N	��DrK��L��IMs%氄,��'a;.�l6M.I�"�F.U��b�K�R�LS�B� fӅLTz��Id$0�Cf�iT62%R�HT,Zr�t��`�Q�XL>	v��	(����5q4F��&�8�:��#��i|�Ch�ۻ�*%������ǟ�`�/U]�Y<�3�w��?�v����������������,����\_�o}i���͇�ܽ}���ӛ�v�$J%���Յ�<oo�5_ID<���|��x솆z��r���R}����������v�թr�u.���и=�U;�R� ;��(�u�V>�L�F ��B�ol����H&��j�٬Sk�`�Z�1�HMv���y,14�������m$j�M�r��GE�S�"���1qT.����x��B�R�Z.[�d�i4/���q#\V�͈rX	>7�f%��(���S:i���i�1�Zk�S�����~�4i���\���:&гh*Uâ�hASCX!��o���9��#\[����g���@'����T���(�����;O���n�/�o��o�NL!� sC��don��=�3�݉�L��;9��I�e3c����D.=�I&�}Ao*-��x�g~b>������핹������n�z���Aw_�;K9Gfl:��������(��}����W��_�I)�K�(��+�88`��_��>������o~r��Ͼ���o�B��Ϸ���_~s�W�o���B)��ן�������>����;���q ��'�?������W׾����7�|����ž��xk���y��O?�:�B��Ԉ�e�x �����������H }�{[�v�6v� ���G���xFM��ѵ8���`M]p��fB�\C]gݥf�b ��W�@�����O&2�)F�:Hh�,�@�܈�P^��gZNG�˖ʥz>WI&r[�D<�͔J�z� �F���qER�>�rKS���z�C��c�|��.��b��a��{�xhfjb)��o1)m^K�'��E{Bf��'C�P8R���c(�d0>28>40:�?1҇T�G��c�C�c3�s ��i����"����
8���*��3룽�>=��Q�7�I���9��@*;ܓHe����,Z�M�w��N��_��O�1��&�抹Q��b�8&:����Ϭ��������?�����<؋�+l�sM`	ҧzlihh�Ա�����n_l�1(K����/�� #+���?�-��Bkq-K<� �Eb�`�H����-腷 ���l��<�h��L�������   IDAT yL$����\�0 ��s��N�����0PߋD���ς�����ӿ'��'If���V�? ���[��ƣo$�$���p��9�@.�φ�,�T���)-)uuF�^�	0�o����ޅ~�?��SpEsycw�鈻\1�3�v�}��Xxj��4�H��J�DL��L�b\S����6���X*O��h �¡�(t����*�Jc�il�.��fq����u��et������-��B`�Z	#�3�[����1p��>��	\ܢYح����������"*JN!��a	G�R�m�ܘ�����lO���wO��J��|�����U!��-ݫc�\7�-�(fQ�A�����]M�v����n$s��\�J�R�͍��V��Lx"�������ɵ�*Ͼ4{�]�卽��[�L��8����������v��w�޿�:\�.��N�?y�����'��O�`��>�}tW�pG|zO�G�uG�v#ck�K3�c|�R%�m���Z񌒯�\�1Of'n��&G��O�e.�;���48T2�eɁi0`"��Wi<v���>`�4�����*=κ1syB��N��D��d��6��.L�����nw�b�X`����o�����l�]</
=.Gw:���c^g�f\-��][�Φ��7&��J��l��}=�\*����	�����@lr(
LF'������`h��M�������@>�{.s�gM�]��?r�b> ��!�w��"y䢡D8 �	A��J���ǂ�)%��;�M��z2���� ���b���ۻ���fr�\ߘ*f���Ǜ�#�ؑl�� z�a�X:�QI[1)������8�E.w��^�f�`��Y�r��%SL60�fO�x��|>ʬP�(��Ȕ+
�!W�I��2�T1)V�	�yQ�	��Pɒ�:�U�J	O<5:-��ѩсt.�L,�G���U�<�-�S��аmǵ�vUU֔:w О𴢢��2Lu�ED�\,Ct���7%�Yq>`�-�*g={��*Z/�p���
X\Vv�NT�/��nt�W��]����m͚�v[G{�B��\�[��mry�b���.S�*,��i���uGН!�$�݉h&I��P��(59�Ч�^���!}�X��Ϋ�Eg�)4�B!�a��"��|&�!��j�B�����XQY������*�t���ڪ��궶f8B�r��`�D�vO %��A�e�X��(4*�I�7���4�����b�'e5�V���V��R�Ql6)���!0��[བྷ�Zpn{ Z0O?��_��7����~�����ӿ�w��w�7�7�����������������o���/����]�*�N"ay<&�M1��k3��=
%��7��a����~��lW�XJC� y*o@�u� ����m4�5�6�os��>~$dp9dN�Ԧ��|������*�#"5��{����o�8���U���Zp`�_�^	��hf������X��@l��}w�x�=�5�΁k��J8F� ب��4\[#�@���`�b�N�u�6�id��j�mi�G�>hzdⱀ�&��J�V.�j�ǬW��-/)Dt�/����<hF	�m�MW�.��|���tZ�z��I%*�@.��)d��$�8`����P�o��.`
��1A�I<x/�ԁ���Q�:=���Il)� �b5Ma`+u���5�	��bJyr0��W�\�(8��+���Ν���S�)��(�����~nb�-����\a�o ��o��e{�fg�'�F��Gp��L
�'���鐯'.��	��G�T*�Ȅ]�`�	_kP�j0[�����䆆��Z4mq��Յ���Zo�!��3J��&x�������uu��]�C%x�B����R���� ����߮f<��B����>*�`�P璘 �/*�
ML�sp1��dҵ���rs9~� 8�a��0�c1�|VR�Ih�!W�$�e<`.�9�a�~���o=JY/��c0rNI!� �y,�_%�ba� ��	A���Nfm��N?��y����;7ܺ~z�ƽ[G T@�Kuā���>=>�w���;/�x������wa������+SJu2�E�Ӌ�Z��a��ܶ(��QI��ߠ���J��a��) ^�:caOʨř����1���k�C�}��n\���ST��sF��܁�|z���/�~���GG�1)Gt�*ɤ�{ex�f�g#�}�j�w?��~x �����B�~�z ����m��_~����ǻ`�?��6
b�]E�����?��ٍO���\;X,>�6��ϟ���kPÕ�RE9:����Fe�N�75t�6�@�s�!1_��FD3N��R̎<\���y��Ne��l�aS�b���p�R�F�W���О&�M(��Tt�ю64Ƭg8�6\2�4��RS9��RU8�A�[��X����x@]"� e.y]_I�J�i���'��4ص8u��o"�g��bi	(+��>Mv�6�I�6�j��`�L'T��������������������������������&�u�V��W���X�][����Ѹ�������|qjixvy`z�wb�02���.��q��g���-�ݎ��V�"r��[�p���Ƌ3 ����"=�K݌�4x�M�+t����Ip�s��;7��Gh�%w>�n�/��pF���ED��A}K���� ��'�BoCL�-L��O��Rm�ό�M��7)�߇����e�oIC�r3�B*��w-.�����V��אܪ�����d>�ɖ�^��,��p����-�����gv]`āK�J���?-�o�H#���[��|4/:XUSTh�o�@8��:Pg�at5x��#	�����
���y�\�a