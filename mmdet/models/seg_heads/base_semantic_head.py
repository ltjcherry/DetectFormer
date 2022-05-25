# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class HeuristicFusionHead(BasePanopticFusionHead):
    """Fusion Head with Heuristic method."""

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(HeuristicFusionHead,
              self).__init__(num_things_classes, num_stuff_classes, test_cfg,
                             None, init_cfg, **kwargs)

    def forward_train(self, gt_masks=None, gt_semantic_seg=None, **kwargs):
        """HeuristicFusionHead has no training loss."""
        return dict()

    def _lay_masks(self, bboxes, labels, masks, overlap_thr=0.5):
        """Lay instance masks to a result map.

        Args:
            bboxes: The bboxes results, (K, 4).
            labels: The labels of bboxes, (K, ).
            masks: The instance masks, (K, H, W).
            overlap_thr: Threshold to determine whether two masks overlap.
                default: 0.5.

        Returns:
            Tensor: The result map, (H, W).
        """
        num_insts = bboxes.shape[0]
        id_map = torch.zeros(
            masks.shape[-2:], device=bboxes.device, dtype=torch.long)
        if num_insts == 0:
            return id_map, labels

        scores, bboxes = bboxes[:, -1], bboxes[:, :4]

        # Sort by score to use heuristic fusion
        order = torch.argsort(-scores)
        bboxes = bboxes[order]
        labels = labels[order]
        segm_masks = masks[order]

        instance_id = 1
        left_labels = []
        for idx in range(bboxes.shape[0]):
            _cls = labels[idx]
            _mask = segm_masks[idx]
            instance_id_map = torch.ones_like(
                _mask, dtype=torch.long) * instance_id
            area = _mask.sum()
            if area == 0:
                continue

            pasted = id_map > 0
            intersect = (_mask * pasted).sum()
            if (intersect / (area + 1e-5)) > overlap_thr:
                continue

            _part = _mask * (~pasted)
            id_map = torch.where(_part, instance_id_map, id_map)
            left_labels.append(_cls)
            instance_id += 1

        if len(left_labels) > 0:
            instance_labels = torch.stack(left_labels)
        else:
            instance_labels = bboxes.new_zeros((0, ), dtype=torch.long)
        assert instance_id == (len(instance_labels) + 1)
        return id_map, instance_labels

    def simple_test(self, det_bboxes, det_labels, mask_preds, seg_preds,
                    **kwargs):
        """Fuse the results of instance and 