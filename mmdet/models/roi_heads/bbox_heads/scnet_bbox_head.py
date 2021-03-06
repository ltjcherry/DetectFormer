B
    ??a?J  ?               @   s?   d dl Z d dlmZ d dlmZmZmZ d dlmZm	Z	 d dl
mZmZ d dlmZ d dlmZmZ d dlmZ d dlmZ d d	lmZ d
dlmZ e?? G dd? de??ZdS )?    N)?bias_init_with_prob?build_activation_layer?build_norm_layer)?FFN?MultiheadAttention)?	auto_fp16?
force_fp32)?multi_apply)?HEADS?
build_loss)?reduce_mean)?accuracy)?build_transformer?   )?BBoxHeadc                   s?   e Zd ZdZdddddddd	ed
dd?eddddded
dd?edd?d?eddd?df? fdd?	Z? fdd?Ze? dd? ?Ze	dd?d%dd ??Z
d!d"? Zd&d#d$?Z?  ZS )'?DIIHeadaq  Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    ?P   ?   ?   r   ?   i   ?   g        ?ReLUT)?type?inplace?DynamicConv?@   ?   ?LN)r   )r   ?in_channels?feat_channels?out_channels?input_feat_shape?act_cfg?norm_cfg?GIoULossg       @)r   ?loss_weightNc                s?  |d kst d??tt| ?jf |dd|d?|?? t|?| _|| _d| _t|||?| _	t
tdd?|?d | _t|
?| _t?|?| _t
tdd?|?d | _t||||	|d?| _t
tdd?|?d | _t?? | _x\t|?D ]P}| j?tj||dd	?? | j?t
tdd?|?