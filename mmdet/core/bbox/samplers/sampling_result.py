# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_match

from ..builder import BBOX_SAMPLERS
from ..transforms import bbox2roi
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class ScoreHLRSampler(BaseSampler):
    r"""Importance-based Sample Reweighting (ISR_N), described in `Prime Sample
    Attention in Object Detection <https://arxiv.org/abs/1904.04821>`_.

    Score hierarchical local rank (HLR) differentiates with RandomSampler in
    negative part. It firstly computes Score-HLR in a two-step way,
    then linearly maps score hlr to the loss weights.

    Args:
        num (int): Total number of sampled RoIs.
        pos_fraction (float): Fraction of positive samples.
        context (:class:`BaseRoIHead`): RoI head that the sampler belongs to.
        neg_pos_ub (int): Upper bound of the ratio of num negative to num
            positive, -1 means no upper bound.
        add_gt_as_proposals (bool): Whether to add ground truth as proposals.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        score_thr (float): Minimum score that a negative sample is to be
            considered as valid bbox.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 k=0.5,
                 bias=0,
                 score_thr=0.05,
                 iou_thr=0.5,
                 **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        self.k = k
        self.bias = bias
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.context = context
        # context of cascade detectors is a list, so distinguish them here.
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.bbox_roi_extractor
            self.bbox_head = context.bbox_head
            self.with_shared_head = context.with_shared_head
            if self.with_shared_head:
                self.shared_head = context.shared_head
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    @staticmethod
    def random_choice(gallery, num):
        """Randomly select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).flatten()
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes,
                    feats=None,
                    img_meta=None,
                    **kwargs):
        """Sample negative samples.

        Score-HLR sampler is done in the following steps:
        1. Take the maximum positive score prediction of each negative samples
            as s_i.
        2. Filter out negative samples whose s_i <= score_thr, the left samples
            are called valid samples.
        3. Use NMS-Match to divide valid samples into different groups,
            samples in the same group will greatly overlap with each other
        4. Rank the matched samples in two-steps to get Score-HLR.
            (1) In the same group, rank samples with their scores.
            (2) In the same score rank across different groups,
                rank samples with their scores again.
        5. Linearly map Score-HLR to the final label weights.

        Args:
            assign_result (:obj:`AssignResult`): result of assigner.
            num_expected (int): Expected number of samples.
            bboxes (Tensor): bbox to be sampled.
            feats (Tensor): Features come from FPN.
            img_meta (dict): Meta information dictionary.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()
        num_neg = neg_inds.size(0)
        if num_neg == 0:
            return neg_inds, None
        with torch.no_grad():
            neg_bboxes = bboxes[neg_inds]
            neg_rois = bbox2roi([neg_bboxes])
            bbox_resu