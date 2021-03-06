B
    ??a?  ?               @   sT   d dl mZmZ d dlmZmZ d dlmZ ddlm	Z	 e?
? G dd? de	??ZdS )	?    )?
ConvModule?Linear)?
ModuleList?	auto_fp16)?HEADS?   )?FCNMaskHeadc                   sb   e Zd ZdZddddededd?edd	d
d?gd?f? fdd?	Z? fdd?Ze? dd? ?Z?  Z	S )?CoarseMaskHeadaW  Coarse mask head used in PointRend.

    Compared with standard ``FCNMaskHead``, ``CoarseMaskHead`` will downsample
    the input feature map instead of upsample it.

    Args:
        num_convs (int): Number of conv layers in the head. Default: 0.
        num_fcs (int): Number of fc layers in the head. Default: 2.
        fc_out_channels (int): Number of output channels of fc layer.
            Default: 1024.
        downsample_factor (int): The factor that feature map is downsampled by.
            Default: 2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    r   ?   i   ?Xavier?