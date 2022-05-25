# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


class Transition(BaseModule):
    """Base class for transition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(x):
        pass


class UpInterpolationConv(Transition):
    """A transition used for up-sampling.

    Up-sample the input by interpolation then refines the feature by
    a convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Up-sampling factor. Default: 2.
        mode (int): Interpolation mode. Default: nearest.
        align_corners (bool): Whether align corners when interpolation.
            Default: None.
        kernel_size (int): Kernel size for the conv. Default: 3.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=2,
                 mode='nearest',
                 align_corners=None,
                 kernel_size=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(in_channels, out_channels, init_cfg)
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            **kwargs)

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        x = self.conv(x)
        return x


class LastConv(Transition):
    """A transition used for refining the output of the last stage.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_inputs (int): Number of inputs of the FPN features.
        kernel_size (int): Kernel size for the conv. Default: 3.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_inputs,
                 kernel_size=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(in_channels, out_channels, init_cfg)
        self.num_inputs = num_inputs
        self.conv_out = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            **kwargs)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        return self.conv_out(inputs[-1])


@NECKS.register_module()
class FPG(BaseModule):
    """FPG.

    Implementation of `Feature Pyramid Grids (FPG)
    <https://arxiv.org/abs/2004.03580>`_.
    This implementation only gives the basic structure stated in the paper.
    But users can implement different type of transitions to fully explore the
    the potential power of the structure of FPG.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        paths (list[str]): Specify the path order of each stack level.
            Each element in the list should be either 'bu' (bottom-up) or
            'td' (top-down).
        inter_channels (int): Number of inter channels.
        same_up_trans