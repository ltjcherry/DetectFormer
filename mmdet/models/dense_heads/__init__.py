B
    ??aHL  ?               @   s?   d dl Z d dlZd dlmZ d dlm  mZ d dlmZ d dl	m
Z
 d dlmZmZ ddlmZmZ ddlmZ d	Ze?? G d
d? de??ZdS )?    N)?Scale)?
force_fp32)?multi_apply?reduce_mean?   )?HEADS?
build_loss?   )?AnchorFreeHeadg    ?חAc                   s?   e Zd ZdZdddddeffdddded	d
dddd?eddd?edd
dd?eddd
d?edddeddddd?d?f
? fdd?	Z? fdd?Zd d!? Z? fd"d#?Z	e
d$d%?d1d'd(??Zd)d*? Zd+d,? Zd-d.? Zd2? fd/d0?	Z?  ZS )3?FCOSHeada?  Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN'