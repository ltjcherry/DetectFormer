# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from ..utils.brick_wrappers import adaptive_avg_pool2d
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class HybridTaskCascadeRoIHead(CascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(