B
    �a�  �               @   s:   d dl Z d dlmZ ddlmZ e�� G dd� d��ZdS )�    N)�build_from_cfg�   )�	PIPELINESc               @   s(   e Zd ZdZdd� Zdd� Zdd� ZdS )	�Composez�Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    c             C   sj   t |tjj�st�g | _xL|D ]D}t |t�rDt|t�}| j�	|� qt
|�rZ| j�	|� qtd��qW d S )Nz$transform must be callable or a dict)�
isinstance�collections�abc�Sequence�AssertionError�
transforms�dictr   r   �append�callable�	TypeError)�selfr   �	transform� r   �=/home/buu/ltj/mmdetection/mmdet/datasets/pipelines/compose.py�__init__   s    


zCompose.__init__c             C   s(   x"| j D ]}||�}|dkrdS qW |S )z�Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        N)r   )r   �data�tr   r   r   �__call__   s
    
zCompose.__call__c             C   s>   | j jd }x$| jD ]}|d7 }|d|� �7 }qW |d7 }|S )N�(�
z    z
))�	__class__�__name__r   )r   �format_stringr   r   r   r   �__repr__.   s    zCompose.__repr__N)r   �
__module__�__qualname__�__doc__r   r   r   r   r   r   r   r   	   s   r   )r   �
mmcv.utilsr   �builderr   �register_moduler   r   r   r   r   �<module>   s                 