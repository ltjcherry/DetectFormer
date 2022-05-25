# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler


@BBOX_SAMPLERS.register_module()
class InstanceBalancedPosSampler(RandomSampler):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds