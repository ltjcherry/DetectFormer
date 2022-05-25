B
    �a�  �               @   sL   d dl mZ d dlmZ d dlmZ ddlmZ e�� G dd� de��Z	dS )�    N)�
ConvModule)�
BaseModule�   )�NECKSc            
       sF   e Zd ZdZdddedd�dedddd	�f� fd
d�	Zdd� Z�  ZS )�ChannelMappera_  Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    �   N�ReLU)�type�Xavier�Conv2d�uniform)r	   �layer�distributionc	                s�   t t| ��|� t|t�st�d | _|d kr4t|�}t�	� | _
x2|D ]*}	| j
�t|	|||d d |||d�� qDW |t|�kr�t�	� | _xPtt|�|�D ]>}
|
t|�kr�|d }	n|}	| j�t|	|ddd|||d�� q�W d S )N�   r   )�padding�conv_cfg�norm_cfg�act_cfg�����r   )�strider   r   r   r   )�superr   �__init__�
isinstance�list�AssertionError�extra_convs�len�nn�
ModuleList�convs�appendr   �range)�self�in_channels�out_channels�kernel_sizer   r   r   Znum_outs�init_cfg�
in_channel�i)�	__class__� �>/home/buu/ltj/mmdetection/mmdet/models/necks/channel_mapper.pyr   .   s@    





zChannelMapper.__init__c                s�   t � �t �j�kst�� �fdd�tt � ��D �}�jr�xPtt �j��D ]>}|dkrn|��jd � d �� qH|��j| |d �� qHW t|�S )zForward function.c                s   g | ]}�j | � | ��qS r*   )r   )�.0r(   )�inputsr"   r*   r+   �
<listcomp>]   s    z)ChannelMapper.forward.<locals>.<listcomp>r   r   )r   r   r   r!   r   r    �tuple)r"   r-   �outsr(   r*   )r-   r"   r+   �forwardZ   s    zChannelMapper.forward)�__name__�
__module__�__qualname__�__doc__�dictr   r1   �__classcell__r*   r*   )r)   r+   r   	   s   ##r   )
�torch.nnr   �mmcv.cnnr   �mmcv.runnerr   �builderr   �register_moduler   r*   r*   r*   r+   �<module>   s
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ��J�  ��IDAT�'Ȩ*�/�_b�T�O%��V�{�M���ܯ��KCk��U�P\�c͕y�f�5R�ҟ��}3.4$-�0�L�y���b���Ӑ��J���g�UX{E�ʰD[F;ڗ_ݢ)�+{��3��@_���U-�E��,�<�f����B~�6�KٿT��b��**K'e�� UŸ�(n��@�H #���(��@�m����}��  /��� ���#�G��# ��%�;�]@�v�� �����_q�;������~d�l���r�6?^Z� %4��� �$} �g�����H@YI��N\Ky~�Pn 6����w��#c�@�b�e p��Ga#�?ڽ}���M��I��v �;�y�caMa�msI
�����M� eD8::���l�pp�Z�p	wO�!� :�Aﶽp���p���  RX@����M�k��jX�ݰ. _`��^����-ޯͳ27�+Lb��׭��b�Je�l�\🷵T���(^�u��[4������ESbI�O�ְl�h�T΂~�v�G@H	�gg ��@�X�W.�("�I)+*_B!	�p�a�HdƼQ	%�)���7�#J���w	��H+��,��'��"LӃ _H�@B� �F�>����}�)ZO���o��=[�?����h��]�RvU�#+�]��.�ԬY�!K�,[X�����tC
[(,Y���2�� ��x-���EQ��0��hem�*��Tx\2@%����R�EAyR�伾ksܜ��:�Y��R�k�JU{ޒ�0�ڔ�;���c�c��y>{P伶��  ��e�:���G�s�־�ZChO%i�����U`�8o�3<G9l���Ɗ9���-���BĒ��E�2 ���\�^��؛�vʺ�Y۴�zӾ�����^���7o� t��Џ Fy(1�_������a�����ߠ����?�b��7_ ^z�K�Z�-�LÌ�����]�����<���S�:9�ӡ�m�\xS��0�*|�� D��J��y��X� ��!c�*PG#A�)��[�{}K�uOZ��w����3O �J.Y泌�e����d>����{@;�ک���q�C��(%:�Q��c ��T8W]��Egd/��Ă(��"qM��{E�\3"/�� ��� P��l=�]����������Zi��e���~���f����s����_i�}��3����O�۟G�+ǿ���(o�*8��Ө�^$���K>�7�2 ��F.�����;v������O#������[���{{��+���