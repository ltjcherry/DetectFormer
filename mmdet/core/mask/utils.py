B
    ??a1?  ?               @   s?   d dl mZmZ d dlZd dlZd dlZd dlmZ	 d dl
Z
d dlmZ G dd? ded?ZG dd? de?ZG d	d
? d
e?Zdd? ZdS )?    )?ABCMeta?abstractmethodN)?	roi_alignc               @   s?   e Zd ZdZed#dd??Zed$dd??Zed%dd	??Zed
d? ?Zedd? ?Z	ed&dd??Z
edd? ?Zeedd? ??Zedd? ?Zedd? ?Zed'dd??Zd(dd?Zed)d!d"??ZdS )*?BaseInstanceMaskszBase class for instance masks.?nearestc             C   s   dS )a]  Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`.

        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`mmcv.imrescale`.

        Returns:
            BaseInstanceMasks: The rescaled masks.
        N? )?self?scale?interpolationr   r   ?7/home/buu/ltj/mmdetection/mmdet/core/mask/structures.py?rescale   s    zBaseInstanceMasks.rescalec             C   s   dS )z?Resize masks to the given out_shape.

        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`mmcv.imresize`.

        Returns:
            BaseInstanceMasks: The resized masks.
        Nr   )r   ?	out_shaper
   r   r   r   ?resize   s    
zBaseInstanceMasks.resize?
horizontalc             C   s   dS )z?Flip masks alone the given direction.

        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.

        Returns:
            BaseInstanceMasks: The flipped masks.
        Nr   )r   ?flip_directionr   r   r   ?flip(   s    	zBaseInstanceMasks.flipc             C   s   dS )z?Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.

        Returns:
            BaseInstanceMasks: The padded masks.
        Nr   )r   r   ?pad_valr   r   r   ?pad3   s    
zBaseInstanceMasks.padc             C   s   dS )z?Crop each mask by the given bbox.

        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).

        Return:
            BaseInstanceMasks: The cropped masks.
        Nr   )r   ?bboxr   r 