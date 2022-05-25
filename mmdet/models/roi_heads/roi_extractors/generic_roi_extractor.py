B
    �a�  �               @   sL   d dl mZ d dlmZ d dlmZ ddlmZ e�� G dd� de��Z	dS )	�    )�build_plugin_layer)�
force_fp32)�ROI_EXTRACTORS�   )�BaseRoIExtractorc                   s8   e Zd ZdZd� fdd�	Zeddd�dd	d
��Z�  ZS )�GenericRoIExtractorag  Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    �sumNc                sh   t t| �jf |� |dkst�|| _|d k	| _|d k	| _| jrNt|d�d | _| jrdt|d�d | _	d S )N)r   �concatZ_post_moduler   Z_pre_module)
�superr   �__init__�AssertionError�aggregation�	with_post�with_prer   �post_module�
pre_module)�selfr   Zpre_cfgZpost_cfg�kwargs)�	__class__� �X/home/buu/ltj/mmdetection/mmdet/models/roi_heads/roi_extractors/generic_roi_extractor.pyr      s    

zGenericRoIExtractor.__init__)�featsT)�apply_to�out_fp16c             C   s  t |�dkr | jd |d |�S | jd j}t |�}|d j|�d�| jf|�� }|jd dkrd|S |dk	rx| �||�}d}xnt|�D ]b}| j| || |�}	||	�d� }
| j	r�| �
|	�}	| jdkr�||	7 }n|	|dd�||
�f< |
}q�W | jdk�r|| jk�st�| j�r| �|�}|S )zForward function.r   r   Nr   r	   )�len�
roi_layers�output_size�	new_zeros�size�out_channels�shape�roi_rescale�ranger   r   r   r   r   r   )r   r   �rois�roi_scale_factor�out_size�
num_levels�	roi_featsZstart_channels�iZroi_feats_tZend_channelsr   r   r   �forward+   s2    



zGenericRoIExtractor.forward)r   NN)N)�__name__�
__module__�__qualname__�__doc__r   r   r)   �__classcell__r   r   )r   r   r   	   s     
r   N)
Zmmcv.cnn.bricksr   �mmcv.runnerr   Zmmdet.models.builderr   �base_roi_extractorr   �register_moduler   r   r   r   r   �<module>   s
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       