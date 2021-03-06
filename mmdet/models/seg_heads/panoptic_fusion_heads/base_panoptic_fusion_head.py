B
    ??az  ?               @   sH   d dl Z d dlmZ d dlmZ ddlmZ e?? G dd? de??ZdS )?    N)?INSTANCE_OFFSET)?HEADS?   )?BasePanopticFusionHeadc                   s>   e Zd ZdZd? fdd?	Zddd?Zdd
d?Zdd? Z?  ZS )?HeuristicFusionHeadz"Fusion Head with Heuristic method.?P   ?5   Nc                s    t t| ?j|||d |f|? d S )N)?superr   ?__init__)?self?num_things_classes?num_stuff_classes?test_cfg?init_cfg?kwargs)?	__class__? ?_/home/buu/ltj/mmdetection/mmdet/models/seg_heads/panoptic_fusion_heads/heuristic_fusion_head.pyr
      s    zHeuristicFusionHead.__init__c             K   s   t ? S )z)HeuristicFusionHead has no training loss.)?dict)r   ?gt_masks?gt_semantic_segr   r   r   r   ?forward_train   s    z!HeuristicFusionHead.forward_train?      ??c             C   sp  |j d }tj|j dd? |jtjd?}|dkr8||fS |dd?df |dd?dd?f  }}t?| ?}|| }|| }|| }	d}
g }x?t|j d ?D ]?}|| }|	| }tj|tjd?|
 }|?? }|dkr?q?|dk}|| ?? }||d	  |kr?q?||  }t?	|||?}|?
|? |
d7 }
q?W t|?dk?rBt?|?}n|jd
tjd?}|
t|?d k?sht?||fS )au  Lay instance masks to a result map.

        Args:
            bboxes: The bboxes results, (K, 4).
            labels: The labels of bboxes, (K, ).
            masks: The instance masks, (K, H, W).
            overlap_thr: Threshold to 