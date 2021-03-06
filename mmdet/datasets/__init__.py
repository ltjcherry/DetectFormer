B
    ??a?  ?               @   s@   d dl Z d dlZd dlmZ d dlmZ G dd? de?ZeZdS )?    N)?COCO)?COCOevalc                   sj   e Zd ZdZd? fdd?	Zg g g dfdd?Zg g g fdd?Zg g fd	d
?Zdd? Zdd? Z	dd? Z
?  ZS )r   z?This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    Nc                s>   t tdd?dkrt?dt? t? j|d? | j| _| j	| _
d S )N?__version__?0z12.0.2z]mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools")?annotation_file)?getattr?pycocotools?warnings?warn?UserWarning?super?__init__Z	imgToAnnsZimg_ann_mapZ	catToImgs?cat_img_map)?selfr   )?	__class__? ?A/home/buu/ltj/mmdetection/mmdet/datasets/api_wrappers/coco_api.pyr      s    zCOCO.__init__c             C   s   | ? ||||?S )N)?	getAnnIds)r   ?img_ids?cat_idsZarea_rng?iscrowdr   r   r   ?get_ann_ids   s    zCOCO.get_ann_idsc             C   s   | ? |||?S )N)Z	getCatIds)r   ?	cat_namesZ	sup_namesr   r   r   r   ?get_cat_ids   s    zCOCO.get_cat_idsc             C   s   | ? ||?S )N)Z	getImgIds)r   r   r