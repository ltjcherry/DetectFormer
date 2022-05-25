B
    �a�  �               @   s�   d dl mZ d dlZd dlZd dlmZmZ ddlm	Z	m
Z
 dd� Zddd	�Zd
d� Zdd� Zddd�Zddd�Zddd�Zddd�ZdS )�    )�partialN)�map�zip�   )�BitmapMasks�PolygonMasksc             O   s4   |rt | f|�n| }t|f|�� }tttt|� ��S )a  Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains             a kind of returned results by the function
    )r   r   �tuple�listr   )�func�args�kwargsZpfuncZmap_results� r   �2/home/buu/ltj/mmdetection/mmdet/core/utils/misc.py�multi_apply   s    r   c             C   sj   | � � dkr,| �|f|�}| ||�tj�< n:|f| �� dd�  }| �||�}| ||�tj�dd�f< |S )zSUnmap a subset of item (data) back to the original set of items (of size
    count)�   N)�dim�new_full�type�torch�bool�size)�data�count�inds�fill�ret�new_sizer   r   r   �unmap!   s    r   c             C   sZ   t | ttf�r| �� } n>t | tj�r6| �� �� �� } n t | t	j
�sVtdt| �� d���| S )z�Convert Mask to ndarray..

    Args:
        mask (:obj:`BitmapMasks` or :obj:`PolygonMasks` or
        torch.Tensor or np.ndarray): The mask to be converted.

    Returns:
        np.ndarray: Ndarray mask of shape (n, h, w) that has been converted
    zUnsupported z
 data type)�
isinstancer   r   �
to_ndarrayr   �Tensor�detach�cpu�numpy�np�ndarray�	TypeErrorr   )�maskr   r   r   �mask2ndarray.   s    

r(   c             C   sh   | j dkst�dddg}||ks$t�|dkr<t�| dg�}n(|dkrTt�| dg�}nt�| ddg�}|S )a$  flip tensor base on flip_direction.

    Args:
        src_tensor (Tensor): input feature map, shape (B, C, H, W).
        flip_direction (str): The flipping direction. Options are
          'horizontal', 'vertical', 'diagonal'.

    Returns:
        out_tensor (Tensor): Flipped tensor.
    �   �
horizontal�vertical�diagonal�   r   )�ndim�AssertionErrorr   �flip)�
src_tensor�flip_directionZvalid_directions�
out_tensorr   r   r   �flip_tensorA   s    
r4   Tc                sT   t �ttf�st�t��}|r8� �fdd�t|�D �}n� �fdd�t|�D �}|S )a2  Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    c                s   g | ]}�| �  � � �qS r   )r!   )�.0�i)�batch_id�mlvl_tensorsr   r   �
<listcomp>n   s    z&select_single_mlvl.<locals>.<listcomp>c                s   g | ]}�| �  �qS r   r   )r5   r6   )r7   r8   r   r   r9   r   s    )r   r	   r   r/   �len�range)r8   r7   r!   �
num_levelsZmlvl_tensor_listr   )r7   r8   r   �select_single_mlvlX   s    r=   c                s�   | |k}| | } t �|�}t||�d��}| jdd�\} }| d|� } ||d|�  }|jdd�\� }	d}
|dk	r�t|t�r�� fdd�|�� D �}
nHt|t	�r�� fd	d
�|D �}
n*t|t j
�r�|�  }
ntdt|�� d���| |	� |
fS )a�  Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered,                 shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape                 (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape                 (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional):                 The filtered results. The shape of each item is                 (num_bboxes_filtered, N).
    r   T)�
descendingNr   )r   c                s   i | ]\}}|�  |�qS r   r   )r5   �k�v)�	keep_idxsr   r   �
<dictcomp>�   s    z*filter_scores_and_topk.<locals>.<dictcomp>c                s   g | ]}|�  �qS r   r   )r5   �result)rA   r   r   r9   �   s    