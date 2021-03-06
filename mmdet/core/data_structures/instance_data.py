B
    οΊa6*  γ               @   s8   d dl Z d dlZd dlZd dlmZ G dd deZdS )ι    N)ΪNiceReprc                   sΨ   e Zd ZdZd/ddZdd Zdd Zd0d	d
Zdd Zdd Z	dd Z
dd Zdd Zdd Z fddZ fddZeZeZdd Zdd Zdd  Zd!d" Zd#d$ Zd%d& Zd'd( Zd)d* Zd+d, Zd-d. Z  ZS )1ΪGeneralDataaZ  A general data structure of OpenMMlab.

    A data structure that stores the meta information,
    the annotations of the images or the model predictions,
    which can be used in communication between components.

    The attributes in `GeneralData` are divided into two parts,
    the `meta_info_fields` and the `data_fields` respectively.

        - `meta_info_fields`: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. All attributes in
          it are immutable once set,
          but the user can add new meta information with
          `set_meta_info` function, all information can be accessed
          with methods `meta_info_keys`, `meta_info_values`,
          `meta_info_items`.

        - `data_fields`: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          `.` , `[]`, `in`, `del`, `pop(str)` `get(str)`, `keys()`,
          `values()`, `items()`. Users can also apply tensor-like methods
          to all obj:`torch.Tensor` in the `data_fileds`,
          such as `.cuda()`, `.cpu()`, `.numpy()`, `device`, `.to()`
          `.detach()`, `.numpy()`

    Args:
        meta_info (dict, optional): A dict contains the meta information
            of single image. such as `img_shape`, `scale_factor`, etc.
            Default: None.
        data (dict, optional): A dict contains annotations of single image or
            model predictions. Default: None.

    Examples:
        >>> from mmdet.core import GeneralData
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = GeneralData(meta_info=img_meta)
        >>> img_shape in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        >>> print(results)
        <GeneralData(

          META INFORMATION
        img_shape: (800, 1196, 3)
        pad_shape: (800, 1216, 3)

          DATA FIELDS
        shape of det_labels: torch.Size([4])
        shape of det_scores: torch.Size([4])

        ) at 0x7f84acd10f90>
        >>> instance_data.det_scores
        tensor([0.0100, 0.1000, 0.2000, 0.3000])
        >>> instance_data.det_labels
        tensor([0, 1, 2, 3])
        >>> instance_data['det_labels']
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in instance_data
        True
        >>> instance_data.img_shape
        (800, 1196, 3)
        >>> 'det_scores' in instance_data
        True
        >>> del instance_data.det_scores
        >>> 'det_scores' in instance_data
        False
        >>> det_labels = instance_data.pop('det_labels', None)
        >>> det_labels
        tensor([0, 1, 2, 3])
        >>> 'det_labels' in instance_data
        >>> False
    Nc             C   s:   t  | _t  | _|d k	r$| j|d |d k	r6|  |‘ d S )N)Ϊ	meta_info)ΪsetΪ_meta_info_fieldsΪ_data_fieldsΪset_meta_infoΪset_data)Ϊselfr   Ϊdata© r   ϊD/home/buu/ltj/mmdetection/mmdet/core/data_structures/general_data.pyΪ__init__W   s    zGeneralData.__init__c             C   sΠ   t |tstd| t |‘}x¨| ‘ D ]\}}|| jkr²t| |}t |tj	t
jfr||k ‘ rjq,q°td| dt| | dqΘ||krq,qΘtd| dt| | dq,| j |‘ || j|< q,W dS )zΣAdd meta information.

        Args:
            meta_info (dict): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
        z meta should be a `dict` but get zimg_meta_info z has been set as z before, which is immutable N)Ϊ
isinstanceΪdictΪAssertionErrorΪcopyΪdeepcopyΪitemsr   ΪgetattrΪtorchΪTensorΪnpΪndarrayΪallΪKeyErrorΪaddΪ__dict__)r
   r   ΪmetaΪkΪvZ	ori_valuer   r   r   r   a   s"    


zGeneralData.set_meta_infoc             C   s>   t |tstd| x | ‘ D ]\}}|  ||‘ q"W dS )zͺUpdate a dict to `data_fields`.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions. Default: None.
        z meta should be a `dict` but get N)r   r   r   r   Ϊ__setattr__)r
   r   r   r    r   r   r   r	      s    zGeneralData.set_datac             C   sB   |   ‘ }| t|  ‘ ‘ |dk	r,| |‘ |dk	r>| |‘ |S )a{  Return a new results with same image meta information.

        Args:
            meta_info (dict, optional): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
            data (dict, optional): A dict contains annotations of image or
                model predictions. Default: None.
        N)Ϊ	__class__r   r   Ϊmeta_info_itemsr	   )r
   r   r   Ϊnew_datar   r   r   Ϊnew   s    


zGeneralData.newc             C   s   dd | j D S )zN
        Returns:
            list: Contains all keys in data_fields.
        c             S   s   g | ]}|qS r   r   )Ϊ.0Ϊkeyr   r   r   ϊ
<listcomp>€   s    z$GeneralData.keys.<locals>.<listcomp>)r   )r
   r   r   r   Ϊkeys   s    zGeneralData.keysc             C   s   dd | j D S )zS
        Returns:
            list: Contains all keys in meta_info_fields.
        c             S   s   g | ]}|qS r   r   )r&   r'   r   r   r   r(   «   s    z.GeneralData.meta_info_keys.<locals>.<listcomp>)r   )r
   r   r   r   Ϊmeta_info_keys¦   s    zGeneralData.meta_info_keysc                s    fdd   ‘ D S )zP
        Returns:
            list: Contains all values in data_fields.
        c                s   g | ]}t  |qS r   )r   )r&   r   )r
   r   r   r(   ²   s    z&GeneralData.values.<locals>.<listcomp>)r)   )r
   r   )r
   r   Ϊvalues­   s    zGeneralData.valuesc                s    fdd   ‘ D S )zU
        Returns:
            list: Contains all values in meta_info_fields.
        c                s   g | ]}t  |qS r   )r   )r&   r   )r
   r   r   r(   Ή   s    z0GeneralData.meta_info_values.<locals>.<listcomp>)r*   )r
   r   )r
   r   Ϊmeta_info_values΄   s    zGeneralData.meta_info_valuesc             c   s&   x |   ‘ D ]}|t| |fV  q
W d S )N)r)   r   )r
 