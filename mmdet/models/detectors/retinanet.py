# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import mmcv
import numpy as np
import torch

from mmdet.core.visualization.image import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

INF = 1e8


@DETECTORS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
        