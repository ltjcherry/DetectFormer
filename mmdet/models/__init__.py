# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import Sequential, load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    r"""Bottleneck for the ResNet backbone in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_.

    This bottleneck allows the users to specify whether to use
    SAC (Switchable Atrous Convolution) and RFP (Recursive Feature Pyramid).

    Args:
         inplanes (int): The number of input channels.
         planes (int): The number of output channels before ex