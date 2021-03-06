B
    :??az[  ?               @   s6  d dl Z d dlZd dlmZ d dlm  mZ d dlmZ d dl	m
Z
 d dlmZmZ ddlmZmZ d dlmZmZmZ d dlmZmZ d d	lmZ d
dlmZ dZe?? G dd? de??ZG dd? dej?ZG dd? dej?Z G dd? dej?Z!G dd? dej?Z"G dd? de!?Z#ddd?Z$G dd? dej?Z%dS )?    N)?Scale)?
force_fp32)?multi_apply?reduce_mean?   )?HEADS?
build_loss)?Conv2d?Linear?build_activation_layer)?FFN?build_positional_encoding)?build_transformer?   )?AnchorFreeHeadg    ?חAc                   s?   e Zd ZdZdddddddeffd	d
d	d	edddddd?eddd?edddd?edddd?edddeddddd?d?f? fdd ?	Z? fd!d"?Zd#d$? Z? fd%d&?Z	e
d'd(?d3d)d*??Zd+d,? Zd-d.? Zd/d0? Zd4? fd1d2?	Z?  ZS )5?NiuHeada?  Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

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
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    ?d   N)??????@   )r   ??   )r   ?   )r   i   i   Fg      ???	FocalLossTg       @g      ??g      ??)?type?use_sigmoid?gamma?alpha?loss_weight?L1Lossg      @)r   r   ?CrossEntropyLoss)r   r   r   ?GN?    )r   ?
num_groups?requires_grad?Normalr	   g{?G?z???conv_cls)r   ?name?std?	bias_prob)r   ?layerr&   ?overridec                s`   || _ || _|| _|| _|	| _t? j||f|
|||d?|?? t|?| _t	|?| _
| j
j| _d S )N)?loss_cls?	loss_bbox?norm_cfg?init_cfg)?regress_ranges?center_sampling?center_sample_radius?norm_on_bbox?centerness_on_reg?super?__init__r   ?loss_centernessr   ?transformer?
embed_dims)?self?num_classes?in_channels?	num_queryr6   r.   r/   r0   r1   r2   r*   r+   r5   r,   r-   ?kwargs)?	__class__? ?>/home/buu/ltj/mmdetection/mmdet/models/dense_heads/niu_head.pyr4   @   s     "

zNiuHead.__init__c                s<   t ? ??  tj| jdddd?| _t?dd? | jD ??| _dS )zInitialize layers of the head.r   ?   )?paddingc             S   s   g | ]}t d ??qS )g      ??)r   )?.0?_r>   r>   r?   ?
<listcomp>w   s    z(NiuHead._init_layers.<locals>.<listcomp>N)	r3   ?_init_layers?nnr	   ?feat_channels?conv_centerness?
ModuleList?strides?scales)r8   )r=   r>   r?   rE   s   s    
zNiuHead._init_layersc             C   s   t | j|| j| j?S )a?  Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,                     each is a 4D-tensor, the channel number is                     num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each                     scale level, each is a 4D-tensor, the channel number is                     num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level,                     each is a 4D-tensor, the channel number is num_points * 1.
        )r   ?forward_singlerK   rJ   )r8   ?featsr>   r>   r?   ?forwardy   s    zNiuHead.forwardc       	         sn   t ? ?|?\}}}}| jr&| ?|?}n
| ?|?}||??? }| jr\t?|?}| jsd||9 }n|?	? }|||fS )a4  Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness                 predictions of input feature maps.
        )
r3   rL   r2   rH   ?floatr1   ?F?relu?training?exp)	r8   ?x?scale?stride?	cls_score?	bbox_pred?cls_feat?reg_feat?
centerness)r=   r>   r?   rL   ?   s    


zNiuHead.forward_single)?
cls_scores?
bbox_preds?centernesses)?apply_toc                 s?  t |?t |?  kr t |?ks&n t?dd? |D ?}?jj||d j|d jd?}	??|	||?\}
}|d ?d?? ?fdd?|D ?}dd? |D ?}dd? |D ?}t?	|?}t?	|?}t?	|?}t?	|
?}t?	|?}t?	? fdd?|	D ??}?j
}|dk||k @ ?? ?d	?}tjt |?tj|d jd?}tt|?d
?}?j|||d?}|| }|| }|| }??|?}tt|?? ?? ?d?}t |?dk?r?|| }?j?||?}?j?||?}?j||||d?}?j|||d?}n|?? }|?? }t|||d?S )aZ  Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        c             S   s   g | ]}|? ? d d? ?qS )?????N)?size)rB   ?featmapr>   r>   r?   rD   ?   s    z NiuHead.loss.<locals>.<listcomp>r   )?dtype?devicec                s&   g | ]}|? d ddd??d? j??qS )r   r   r@   r   r   )?permute?reshape?cls_out_channels)rB   rW   )r8   r>   r?   rD   ?   s   c             S   s$   g | ]}|? d ddd??dd??qS )r   r   r@   r   r   ?   )re   rf   )rB   rX   r>   r>   r?   rD   ?   s   c             S   s"   g | ]}|? d ddd??d??qS )r   r   r@   r   r   )re   rf   )rB   r[   r>   r>   r?   rD   ?   s   c                s   g | ]}|? ? d ??qS )r   )?repeat)rB   ?points)?num_imgsr>   r?   rD   ?   s    r   g      ??)?
avg_factorg?????ư>)?weightrl   )r*   r+   r5   )?len?AssertionError?prior_generator?grid_priorsrc   rd   ?get_targetsra   ?torch?catr9   ?nonzerorf   ?tensorrO   ?maxr   r*   ?centerness_target?sum?detach?
bbox_coder?decoder+   r5   ?dict) r8   r\   r]   r^   ?	gt_bboxes?	gt_labels?	img_metas?gt_bboxes_ignore?featmap_sizes?all_level_points?labels?bbox_targets?flatten_cls_scores?flatten_bbox_preds?flatten_centerness?flatten_labels?flatten_bbox_targets?flatten_points?bg_class_ind?pos_inds?num_posr*   ?pos_bbox_preds?pos_centerness?pos_bbox_targets?pos_centerness_targets?centerness_denorm?
pos_points?pos_decoded_bbox_preds?pos_decoded_target_predsr+   r5   r>   )rk   r8   r?   ?loss?   sn    &







zNiuHead.lossc                s  t ??t ?j?kst?t ??}??fdd?t|?D ?}tj|dd?}tj?dd?}dd? ?D ??t?j||||?d?\}}	?fdd?|D ?}?fdd?|	D ?}	g }
g }xdt|?D ]X? |
?t?? fd	d?|D ??? t?? fd
d?|	D ??}?j	r?|?j
?   }|?|? q?W |
|fS )a?  Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.                 concat_lvl_bbox_targets (list[Tensor]): BBox targets of each                     level.
        c                s.   g | ]&}? | ? ?j| ?d  ?? | ??qS )N)?
new_tensorr.   ?	expand_as)rB   ?i)rj   r8   r>   r?   rD   *  s   z'NiuHead.get_targets.<locals>.<listcomp>r   )?dimc             S   s   g | ]}|? d ??qS )r   )ra   )rB   ?centerr>   r>   r?   rD   2  s    )rj   r.   ?num_points_per_lvlc                s   g | ]}|? ? d ??qS )r   )?split)rB   r?   )?
num_pointsr>   r?   rD   >  s    c                s   g | ]}|? ? d ??qS )r   )r?   )rB   r?   )r?   r>   r?   rD   @  s   c                s   g | ]}|?  ?qS r>   r>   )rB   r?   )r?   r>   r?   rD   I  s    c                s   g | ]}|?  ?qS r>   r>   )rB   r?   )r?   r>   r?   rD   K  s    )rn   r.   ro   ?rangers   rt   r   ?_get_target_single?appendr1   rJ   )r8   rj   ?gt_bboxes_list?gt_labels_list?
num_levels?expanded_regress_ranges?concat_regress_ranges?concat_points?labels_list?bbox_targets_list?concat_lvl_labels?concat_lvl_bbox_targetsr?   r>   )r?   r?   rj   r8   r?   rr     s8    
zNiuHead.get_targetsc       (      C   sZ  |? d?}|? d?}|dkr:|?|f| j?|?|df?fS |dd?df |dd?df  |dd?df |dd?df   }|d ?|d?}|dd?ddd?f ?||d?}|d ?||d?}|dd?df |dd?df  }	}
|	dd?df ?||?}	|
dd?df ?||?}
|	|d  }|d |	 }|
|d	  }|d
 |
 }t?||||fd?}| j?r?| j	}|d |d  d }|d	 |d
  d }t?
|?}|?|j?}d}x8t|?D ],\}}|| }| j| | |||?< |}?q?W || }|| }|| }|| }t?||d k||d ?|d< t?||d	 k||d	 ?|d	< t?||d k|d |?|d< t?||d
 k|d
 |?|d
< |	|d  }|d |	 }|
|d	  }|d
 |
 } t?|||| fd?}!|!?d?d dk}"n|?d?d dk}"|?d?d }#|#|d k|#|d	 k@ }$t||"dk< t||$dk< |jdd?\}%}&||& }'| j|'|%tk< |t|?|&f }|'|fS )zACompute regression and classification targets for a single image.r   rh   Nr   r@   r   ).r   ).r   ).r   ).r@   r   )r?   )ra   ?new_fullr9   ?	new_zerosri   ?expandrs   ?stackr/   r0   ?
zeros_like?shape?	enumeraterJ   ?where?minrw   ?INFr?   )(r8   r~   r   rj   r.   r?   r?   ?num_gts?areas?xs?ys?left?right?top?bottomr?   ?radius?	center_xs?	center_ys?
center_gtsrV   ?	lvl_begin?lvl_idx?num_points_lvl?lvl_end?x_mins?y_mins?x_maxs?y_maxs?cb_dist_left?cb_dist_right?cb_dist_top?cb_dist_bottom?center_bbox?inside_gt_bbox_mask?max_regress_distance?inside_regress_range?min_area?min_area_indsr?   r>   r>   r?   r?   Q  sx    

"
"


zNiuHead._get_target_singlec             C   s?   |dd?ddgf }|dd?ddgf }t |?dkr>|d }n@|jdd?d |jdd?d  |jdd?d |jdd?d   }t?|?S )	z?Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        Nr   r   r   r@   ).r   r   )r?   )rn   r?   rw   rs   ?sqrt)r8   r?   ?
left_right?
top_bottom?centerness_targetsr>   r>   r?   rx   ?  s    
"zNiuHead.centerness_targetc       	         sR   t ?d? t? ?||||?\}}tj|?d?| |?d?| fdd?|d  }|S )zbGet points according to feature map size.

        This function will be deprecated soon.
        z?`_get_points_single` in `FCOSHead` will be deprecated soon, we support a multi level point generator nowyou can get points of a single level feature map with `self.prior_generator.single_level_grid_priors` r   )r?   r   )?warnings?warnr3   ?_get_points_singlers   r?   rf   )	r8   ?featmap_sizerV   rc   rd   ?flatten?yrT   rj   )r=   r>   r?   r?   ?  s    
zNiuHead._get_points_single)N)F)?__name__?
__module__?__qualname__?__doc__r?   r}   r4   rE   rN   rL   r   r?   rr   r?   rx   r?   ?__classcell__r>   r>   )r=   r?   r      sN   *
_=Tr   c                   s$   e Zd Z? fdd?Zdd? Z?  ZS )?TransformerLayerc                sx   t ? ??  tj||dd?| _tj||dd?| _tj||dd?| _tj||d?| _tj||dd?| _	tj||dd?| _
d S )NF)?bias)?	embed_dim?	num_heads)r3   r4   rF   r
   ?q?k?v?MultiheadAttention?ma?fc1?fc2)r8   ?cr?   )r=   r>   r?   r4   ?  s    
zTransformerLayer.__init__c             C   s@   | ? | ?|?| ?|?| ?|??d | }| ?| ?|??| }|S )Nr   )r?   r?   r?   r?   r?   r?   )r8   rT   r>   r>   r?   rN   ?  s    (zTransformerLayer.forward)r?   r?   r?   r4   rN   r?   r>   r>   )r=   r?   r?   ?  s   	r?   c                   s$   e Zd Z? fdd?Zdd? Z?  ZS )?TransformerBlockc                s\   t ? ??  d | _|? kr$t|? ?| _t?? ? ?| _tj? ?fdd?t|?D ?? | _	? | _
d S )Nc             3   s   | ]}t ? ??V  qd S )N)r?   )rB   rC   )?c2r?   r>   r?   ?	<genexpr>?  s    z,TransformerBlock.__init__.<locals>.<genexpr>)r3   r4   ?conv?ConvrF   r
   ?linear?
Sequentialr?   ?trr?   )r8   ?c1r?   r?   ?
num_layers)r=   )r?   r?   r?   r4   ?  s    
 zTransformerBlock.__init__c             C   sb   | j d k	r| ? |?}|j\}}}}|?d??ddd?}| ?|| ?|? ??ddd??|| j||?S )Nr   r   r   )r?   r?   r?   re   r?   r?   rf   r?   )r8   rT   ?brC   ?w?h?pr>   r>   r?   rN   ?  s
    

zTransformerBlock.forward)r?   r?   r?   r4   rN   r?   r>   r>   )r=   r?   r?   ?  s   	r?   c                   s&   e Zd Zd? fdd?	Zdd? Z?  ZS )	?C3r   T?      ??c                sn   t ? ??  t|| ?? t|? dd?| _t|? dd?| _td?  |d?| _tj? ??fdd?t	|?D ?? | _
d S )Nr   r   c             3   s    | ]}t ? ? ??d d?V  qdS )g      ??)?eN)?
Bottleneck)rB   rC   )?c_?g?shortcutr>   r?   r?   ?  s    zC3.__init__.<locals>.<genexpr>)r3   r4   ?intr?   ?cv1?cv2?cv3rF   r?   r?   ?m)r8   r?   r?   ?nr  r  r   )r=   )r  r  r  r?   r4   ?  s    
zC3.__init__c             C   s*   | ? tj| ?| ?|??| ?|?fdd??S )Nr   )r?   )r  rs   rt   r	  r  r  )r8   rT   r>   r>   r?   rN   ?  s    z
C3.forward)r   Tr   r?   )r?   r?   r?   r4   rN   r?   r>   r>   )r=   r?   r?   ?  s   	r?   c                   s.   e Zd Zd
? fdd?	Zdd? Zdd	? Z?  ZS )r?   r   NTc          	      sd   t ? ??  tj||||t||?|dd?| _t?|?| _|dkrFt?? nt	|tj
?rV|nt?? | _d S )NF)?groupsr?   T)r3   r4   rF   r	   ?autopadr?   ?BatchNorm2d?bn?SiLU?
isinstance?Module?Identity?act)r8   r?   r?   r?   ?sr?   r  r  )r=   r>   r?   r4     s    
 zConv.__init__c             C   s   | ? | ?| ?|???S )N)r  r  r?   )r8   rT   r>   r>   r?   rN   	  s    zConv.forwardc             C   s   | ? | ?|??S )N)r  r?   )r8   rT   r>   r>   r?   ?forward_fuse  s    zConv.forward_fuse)r   r   Nr   T)r?   r?   r?   r4   rN   r  r?   r>   r>   )r=   r?   r?     s   r?   c                   s   e Zd Zd? fdd?	Z?  ZS )?C3TRr   T?      ??c                s6   t ? ?||||||? t|| ?}t||d|?| _d S )Nrh   )r3   r4   r  r?   r	  )r8   r?   r?   r
  r  r  r   r  )r=   r>   r?   r4     s    zC3TR.__init__)r   Tr   r  )r?   r?   r?   r4   r?   r>   r>   )r=   r?   r    s   r  c             C   s,   |d kr(t | t?r| d ndd? | D ?}|S )Nr   c             S   s   g | ]}|d  ?qS )r   r>   )rB   rT   r>   r>   r?   rD     s    zautopad.<locals>.<listcomp>)r  r  )r?   r?   r>   r>   r?   r    s     r  c                   s&   e Zd Zd? fdd?	Zdd? Z?  ZS )	r  Tr   ?      ??c                sL   t ? ??  t|| ?}t||dd?| _t||dd|d?| _|oD||k| _d S )Nr   r@   )r  )r3   r4   r  r?   r  r  ?add)r8   r?   r?   r  r  r   r  )r=   r>   r?   r4     s
    
zBottleneck.__init__c             C   s*   | j r|| ?| ?|?? S | ?| ?|??S )N)r  r  r  )r8   rT   r>   r>   r?   rN   %  s    zBottleneck.forward)Tr   r  )r?   r?   r?   r4   rN   r?   r>   r>   )r=   r?   r    s   r  )N)&r?   rs   ?torch.nnrF   Ztorch.nn.functional?
functionalrP   ?mmcv.cnnr   ?mmcv.runnerr   ?
mmdet.corer   r   ?builderr   r   r	   r
   r   ?mmcv.cnn.bricks.transformerr   r   Zmmdet.models.utilsr   ?anchor_free_headr   r?   ?register_moduler   r  r?   r?   r?   r?   r  r  r  r>   r>   r>   r?   ?<module>   s0      >
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    