B
    ??a?Z  ?               @   sh   d dl mZmZ d dlZd dlmZ d dlmZ d dlm	Z	m
Z
 d dlmZmZ G dd? de	ed	?ZdS )
?    )?ABCMeta?abstractmethodN)?constant_init)?batched_nms)?
BaseModule?
force_fp32)?filter_scores_and_topk?select_single_mlvlc                   s?   e Zd ZdZd? fdd?	Z? fdd?Zedd? ?Zed	d
?ddd??Z	ddd?Z
ddd?Zddd?Zddd?Zed	d
?ddd??Z?  ZS ) ?BaseDenseHeadzBase class for DenseHeads.Nc                s   t t| ??|? d S )N)?superr
   ?__init__)?self?init_cfg)?	__class__? ?E/home/buu/ltj/mmdetection/mmdet/models/dense_heads/base_dense_head.pyr      s    zBaseDenseHead.__init__c                s:   t t| ???  x&| ?? D ]}t|d?rt|jd? qW d S )N?conv_offsetr   )r   r
   ?init_weights?modules?hasattrr   r   )r   ?m)r   r   r   r      s    
zBaseDenseHead.init_weightsc             K   s   dS )zCompute losses of the head.Nr   )r   ?kwargsr   r   r   ?loss   s    zBaseDenseHead.loss)?
cls_scores?
bbox_preds)?apply_toFTc          
      s?   t ? ?t |?kst?|dkr"d}	nd}	t ? ?t |?ks:t?t ? ?}
? fdd?t|
?D ?}| jj|? d j? d jd?}g }xxtt |??D ]h}|| }t? |?}t||?}|	r?t||?}ndd? t|
?D ?}| j||||||||f|?}|?	|? q?W |S )	a?  Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        NFTc                s   g | ]}? | j d d? ?qS )?????N)?shape)?.0?i)r   r   r   ?
<listcomp>U   s    z,BaseDenseHead.get_bboxes.<locals>.<listcomp>r   )?dtype?devicec             S   s   g | ]}d ?qS )Nr   )r   ?_r   r   r   r    d   s    )
?len?AssertionError?range?prior_generator?grid_priorsr!   r"   r	   ?_get_bboxes_single?append)r   r   r   ?score_factors?	img_metas?cfg?rescale?with_nmsr   ?with_score_factors?
num_levels?featmap_sizes?mlvl_priorsZresult_list?img_id?img_meta?cls_score_list?bbox_pred_list?score_factor_list?resultsr   )r   r   ?
get_bboxes   s2    *


zBaseDenseHead.get_bboxesc	          	   K   s?  |d dkrd}
nd}
|dkr$| j n|}|d }|?dd?}g }g }g }|
rRg }nd}?x@tt||||??D ?](\}\}}}}|?? dd? |?? dd? ks?t?|?d	d
d??dd?}|
r?|?d	d
d??d??? }|?d	d
d??d| j	?}| j
r?|?? }n|?d?dd?dd?f }t||j|t||d??}|\}}}}|d }|d }|
?rX|| }| jj|||d?}|?|? |?|? |?|? |
rl|?|? qlW | j||||d ||||f|	?S )aw  Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape                     [num_bboxes, 5], where the first 4 columns are bounding                     box positions (tl_x, tl_y, br_x, br_y) and the 5-th                     column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding                     box with shape [num_bboxes].
        r   NFT?	img_shape?nms_pre?????r   ?   ?   ?   )?	bbox_pred?priorsrA   rB   )?	max_shape?scale_factor)?test_cfg?get?	enumerate?zip?sizer%   ?permute?reshape?sigmoid?cls_out_channels?use_sigmoid_cls?softmaxr   ?	score_thr?dict?
bbox_coder?decoder*   ?_bbox_post_process)r   r6   r7   r8   r3   r5   r-   r.   r/   r   r0   r;   r<   ?mlvl_bboxes?mlvl_scores?mlvl_labels?mlvl_score_factors?	level_idx?	cls_scorerA   Zscore_factorrB   ?scoresr9   ?labels?	keep_idxs?filtered_results?bboxesr   r   r   r)   m   sV    2$





z BaseDenseHead._get_bboxes_singlec	             K   s?   t |?t |?  kr t |?ks&n t?t?|?}|rB||?|? }t?|?}t?|?}|dk	rpt?|?}|| }|r?|?? dkr?t?||dd?df gd?}
|
|fS t||||j?\}
}|
d|j? }
|| d|j? }|
|fS |||fS dS )aJ  bbox post-processing method.

        The boxes