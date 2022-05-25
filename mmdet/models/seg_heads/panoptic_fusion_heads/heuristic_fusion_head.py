B
    �a  �               @   sh   d dl Z d dlZd dlmZ d dlmZ ddlmZ ddlm	Z	 ddl
mZ e�� G dd	� d	e��ZdS )
�    N)�
ModuleList�   )�HEADS)�ConvUpsample�   )�BaseSemanticHeadc                   sx   e Zd ZdZdddddddddded	d
dd�dedddd�f� fdd�	Zdd� Z� fdd�Z� fdd�Zdd� Z	�  Z
S )�PanopticFPNHeada�  PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.

    Arg:
        num_things_classes (int): Number of thing classes. Default: 80.
        num_stuff_classes (int): Number of stuff classes. Default: 53.
        num_classes (int): Number of classes, including all stuff
            classes and one thing class. This argument is deprecated,
            please use ``num_things_classes`` and ``num_stuff_classes``.
            The module will automatically infer the num_classes by
            ``num_stuff_classes + 1``.
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        fg_range (tuple): Range of the foreground classes. It starts
            from ``0`` to ``num_things_classes-1``. Deprecated, please use
             ``num_things_classes`` directly.
        bg_range (tuple): Range of the background classes. It starts
            from ``num_things_classes`` to ``num_things_classes +
            num_stuff_classes - 1``. Deprecated, please use
            ``num_stuff_classes`` and ``num_things_classes`` directly.
        conv_cfg (dict): Dictionary to construct and config
            conv layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Use ``GN`` by default.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        loss_seg (dict): the loss of the semantic head.
    �P   �5   N�   �   r   �   �GN�    T)�type�
num_groups�requires_grad�CrossEntropyLoss�����g      �?)r   �ignore_index�loss_weightc                s.  |d k	r"t �d� ||d ks"t�tt| ��|d ||� || _|| _|d k	r�|	d k	r�|| _|	| _	|d |d  d | _|	d |	d  d | _t �d| j� d| j� d�� || _
|| _|| | _|| _t� | _xHt||�D ]:}| j�t|||dkr�|nd|dk�r|nd|
|d�� q�W t�|| jd�| _d S )Nz�`num_classes` is deprecated now, please set `num_stuff_classes` directly, the `num_classes` will be set to `num_stuff_classes + 1`r   r   zN`fg_range` and `bg_range` are deprecated now, please use `num_things_classes`=z and `num_stuff_classes`=z	 instead.)�
num_layers�num_upsample�conv_cfg�norm_cfg)�warnings�warn�AssertionError�superr   �__init__�num_things_classes�num_stuff_classes�fg_range�bg_range�start_level�	end_level�
num_stages�inner_channelsr   �conv_upsample_layers�range�appendr   �nn�Conv2d�num_classes�conv_logits)�selfr    r!   r-   �in_channelsr'   r$   r%   r"   r#   r   r   �init_cfg�loss_seg�i)�	__class__� �E/home/buu/ltj/mmdetection/mmdet/models/seg_heads/panoptic_fpn_head.pyr   4   s<    
zPanopticFPNHead.__init__c             C   sf   |� � }|| jk }|| jk|| j| j k  }t�|�}t�||| j |�}t�||� � | j |�}|S )z�Merge thing classes to one class.

        In PanopticFPN, the background labels will be reset from `0` to
        `self.num_stuff_classes-1`, the foreground labels will be merged to
        `self.num_stuff_classes`-th channel.
        )�intr    r!   �torch�clone�where)r/   �gt_semantic_segZfg_maskZbg_maskZ
new_gt_segr5   r5   r6   �_set_things_to_voidl   s    

z#PanopticFPNHead._set_things_to_voidc                s   | � |�}t� �||�S )zjThe loss of PanopticFPN head.

        Things classes will be merged to one class in PanopticFPN.
        )r<   r   �loss)r/   �	seg_predsr;   )r4   r5   r6   r=   �   s    
zPanoptic