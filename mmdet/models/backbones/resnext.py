# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.utils import _pair

from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES


class TridentConv(BaseModule):
    """Trident Convolution Module.

    Args:
        in_channels (int): Number of channels in input.
        out_channels (int): Number of channels in output.
        kernel_size (int): Size of convolution kernel.
        stride (int, optional): Convolution stride. Default: 1.
        trident_dilations (tuple[int, int, int], optional): Dilations of
            different trident branch. Default: (1, 2, 3).
        test_branch_idx (int, optional): In inference, all 3 branches will
            be used if `test_branch_idx==-1`, otherwise only branch with
            index `test_branch_idx` will be used. Default: 1.
        bias (bool, optional): Whether to use bias in convolution or not.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 trident_dilations=(1, 2, 3),
                 test_branch_idx=1,
                 bias=False,
                 init_cfg=None):
        super(TridentConv, self).__init__(init_cfg)
        self.num_branch = len(trident_dilations)
        self.with_bias = bias
        self.test_branch_idx = test_branch_idx
        self.stride = _pair(stride)
        self.kernel_size = _pair(kernel_size)
        self.paddings = _pair(trident_dilations)
        self.dilations = trident_dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def extra_repr(self):
        tmpstr = f'in_channels={self.in_channels}'
        tmpstr += f', out_channels={self.out_channels}'
        tmpstr += f', kernel_size={self.kernel_size}'
        tmpstr += f', num_branch={self.num_branch}'
        tmpstr += f', test_branch_idx={self.test_branch_idx}'
        tmpstr += f', stride={self.stride}'
        tmpstr += f', paddings={self.paddings}'
        tmpstr += f', dilations={self.dilations}'
        tmpstr += f', bias={self.bias}'
        return tmpstr

    def forward(self, inputs):
        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input, self.weight, self.bias, self.stride, padding,
                         dilation) for input, dilation, padding in zip(
                             inputs, self.dilations, self.paddings)
            ]
        else:
            assert len(inputs) == 1
            outputs = [
                F.conv2d(inputs[0], self.weight, self.bias, self.stride,
                         self.paddings[self.test_branch_idx],
                         self.dilations[self.test_branch_idx])
            ]

        return outputs


# Since TridentNet is defined over ResNet50 and ResNet101, here we
# only support TridentBottleneckBlock.
class TridentBottleneck(Bottleneck):
    """BottleBlock for TridentResNet.

    Args:
        trident_dilations (tuple[int, int, int]): Dilations of different
            trident branch.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
        concat_output (bool): Whether to concat the output list to a Tensor.
            `True` only in the last Block.
    """

    def __init__(self, trident_dilations, test_branch_idx, concat_output,
                 **kwargs):

        super(TridentBottleneck, self).__init__(**kwargs)
        self.trident_dilations = trident_dilations
        self.num_branch = len(trident_dilations)
        self.concat_output = concat_output
        self.test_branch_idx = test_branch_idx
        self.conv2 = TridentConv(
            self.planes,
            self.planes,
            kernel_size=3,
            stride=self.conv2_stride,
            bias=False,
            trident_dilations=self.trident_dilations,
            test_branch_idx=test_branch_idx,
            init_cfg=dict(
                type='Kaiming',
                distribution='uniform',
                mode='fan_in',
                override=dict(name='conv2')))

    def forward(self, x):

        def _inner_forward(x):
            num_branch = (
                self.num_branch
                if self.training or self.test_branch_idx == -1 else 1)
            identity = x
            if not isinstance(x, list):
                x = (x, ) * num_branch
                identity = x
                if self.downsample is not None:
                    identity = [self.downsample(b) for b in x]

            out = [self.conv1(b) for b in x]
            out = [self.norm1(b) for b in out]
            out = [self.relu(b) for b in out]

            if self.with_plugins:
                for k in range(len(out)):
                    out[k] = self.forward_plugin(out[k],
                                                 self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = [self.norm2(b) for b in out]
            out = [self.relu(b) for b in o