# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from torch import distributed as dist


@HOOKS.register_module()
class SyncRandomSizeHook(Hook):
    """Change and synchronize the random image size across ranks.
    SyncRandomSizeHook is deprecated, please use Resize pipeline to achieve
    similar functions. Such as `dict(type='Resize', img_scale=[(448, 448),
    (832, 832)], multiscale_mode='range', keep_ratio=True)`.

    Note: Due to the multi-process dataloader, its behavior is different
    from YOLOX's official implementation, the official is to change the
    size every fixed iteration interval and what we achieved is a fixed
    epoch interval.

    Args:
        ratio_range (tuple[int]): Random ratio range. It will be multiplied
            by 32, and then change the dataset output image size.
            Default: (14, 26).
        img_scale (tuple[int]): Size of input image. Default: (640, 640).
        interval (int): The epoch interval of change image size. Default: 1.
        device (torch.device | str): device for returned tensors.
            Default: 'cuda'.
    """

    def __init__(self,
                 ratio_range=(14, 26),
                 img_scale=(640, 640),
                 interval=1,
                 device='cuda'):
        warnings.warn('DeprecationWarning: SyncRandomSizeHook is deprecated. '
                      'Please use Resize pipeline to achieve similar '
                      'functions. Due to the multi-process dataloader, '
                      'its behavior is different from YOLOX\'s official '
                      'implementation, the official is to change the size '
                      'every 