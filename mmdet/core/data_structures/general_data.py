# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import numpy as np
import torch

from .general_data import GeneralData


class InstanceData(GeneralData):
    """Data structure for instance-level annnotations or predictions.

    Subclass of :class:`GeneralData`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501

    Examples:
        >>> from mmdet.core import InstanceData
        >>> import numpy as np
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> results = InstanceData(img_meta)
        >>> img_shape in results
        True
        >>> results.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> results["det_scores"] = torch.Tensor([0.01, 0.7, 0.6, 0.3])
        >>> results["det_masks"] = np.ndarray(4, 2, 2)
        >>> len(results)
        4
        >>> print(resutls)
        <InstanceData(

            META INFORMATION
        pad_shape: (800, 1216, 3)
        img_shape: (800, 1196, 3)

            PREDICTIONS
        shape of det_labels: torch.Size([4])
        shape of det_masks: (4, 2, 2)
        shape of det_scores: torch.Size([4])

        ) at 0x7fe26b5ca990>
        >>> sorted_results = results[results.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.0100, 0.3000, 0.6000, 0.7000])
        >>> sorted_results.det_labels
        tensor([0, 3, 2, 1])
        >>> print(results[results.scores > 0.5])
        <InstanceData(

            META INFORMATION
        pad_shape: (800, 1216, 3)
        img_shape: (800, 1196, 3)

            PREDICTIONS
        shape of det_labels: torch.Size([2])
        shape of det_masks: (2, 2, 2)
        shape of det_scores: torch.Size([2])

        ) at 0x7fe26b6d7790>
        >>> results[results.det_scores > 0.5].det_labels
        tensor([1, 2])
        >>> results[results.det_scores > 0.5].det_scores
        tensor([0.7000, 0.6000])
    """

    def __setattr__(self, name, value):

        if name in ('_meta_info_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray, list)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray, list)}'

            if self._data_fields:
                assert len(value) == len(self), f'the length of ' \
                                             f'values {len(value)} is ' \
                                             f'not consistent with' \
                                             f' the length ' \
                                             f'of this :obj:`InstanceData` ' \
                                             f'{len(self)} '
            super().__setattr__(name, value)

    def __getitem__(self, item):
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceData`: Corresponding values.
        """
        assert len(self), ' This is a empty instance'

        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.BoolTensor))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.new()
        if isinstance(item, (torch.Tensor)):
            assert item.dim() == 1, 'Only support to get the' \
                                 ' values along the first dimension.'
            if isinstance(item, torch.BoolTensor):
                assert len(item) == len(self), f'The shape of the' \
                                               f' input(BoolTensor)) ' \
                                               f'{len(item)} ' \
                                               f' does not match the shape ' \
                                               f'of the indexed tensor ' \
                                               f'in results_filed ' \
                                               f'{len(self)} at ' \
                                               f'first dimension. '

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    # convert to indexes from boolTensor
                    if isinstance(item, torch.BoolTensor):
                        indexes = torch.nonzero(item).view(-1)
                    else:
                        indexes = item
                    for index in indexes:
                        r_list.append(v[index])
                    new_data[k] = r_list
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data

    @staticmethod
    def cat(instances_list):
        """Concat the predictions of all :obj:`InstanceData` in the list.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        new_data = instances_list[0].new()
        for k in instances_list[0]._data_fields:
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            else:
                raise ValueError(
                    f'Can not concat the {k} which is a {type(v0)}')
            new_data[k] = values
        return new_data

    def __len__(self):
        if len(self._data_fields):
            for v in self.values():
                return len(v)
        else:
            raise AssertionError('This is an empty `InstanceData`.')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  �]�J!�\�Y?{�$�۸��x�_EaG9:�5ܸ���|�s�@�\��޹!��ܔ������Sx[p�2x���Wy��ZB=�ϝ���Cu���f8���bh�	>��>���D���0����h1�yw:],�5�9*6���Mx}~8�Z���P|�'��Nc�Ğ��x����L�_�`EQ,Ǡ�G��t��+0>�(.��ᴴ�s�8��x�,��{r�p[�����֒���[]�w�<$����h-�-33�2픩ICt�K��]27Ӄ�0�����Q�tiB���Y��[���E�r9*�/M��B��Q�\��+�Qem~L�{ev�鑉����XwXF-#]m2���p{H��M2A��������uz���m�p �`oD��jEǨI��Q7U���Ps�D�>jn	�E���̡oZ����	g���8_�t���v��/t�4,������:�mK���y��(z/R)�b��(Z.� ����>H�-�������gf�Z�H�ݷ��È����YYs��F��!ՁhYH���hX��#8��u�!m���?8��5�wK�G��Or�� �:O�ą����yH�)��x^�Y��	�Ǫ�����\�p˻|�݃X��K���>��8fv�C�+�|l����u_�6�O�:���^��������:��/C
�����	g�^:u"||�|�r����Q~��sԿW��	���^$��L���b=#!����Si��ZG��<�܊@��}���岄^��_»�W��D�o! ��;�hGSSD�-�kx(�s�9�d䦖A��m�J$ `����.�����FR��z�ݺ/��%Rp:��΄��U# ��%ljy5�./����0?;-�c�01:bֆ1I����%p��#��p��I�L�L� ���W^2)���>#�N��b�"LևXD��c�G �v͍�ɵ�I7��n$^i��`hV݀�&�]τ}xi��[i:#�:�d�.ɥ��y�K��Ļ4P�4��`_o֧��+"�%i�#ʴ]a@2�Y�u� �{U����ݒ+q��;$cw��ޮ6���
U;���S{�5�L(�J۹v/�?9�s�{�J�-z m]����<����{t��=#�  o�~n~q�@B�]�������I^O�:cي;5�GF�4�g���8I���4�����m�����+�@��c]�?~]j��]��IM��ŏ���_��]���Sg�gĞ��Y;1�C��G����N8��&.`�r3��I�DdY�>(�AH�eه"�HK��q�ו��3�7^�/'�l���Pm��y�q�_�8�u����v;~����������'��v��z�ru+��b�|������Nb⊙-&��~x�t��C}X/���H��ì�6�K�i��W�cTU�F�fP�b����$X1Re�d֖���]} �۔$D�S5$��c.�����w�Ps�s�m�&v_���09��F�ª���2�<Z��n��R!��|��z��J�ܖB�*�unB��0.����I\�(�����@B�	��5:�5H�mD�}X��^C�&),�$d�5ib���5,�"��}�H
����ň�!d �2b��Zu��H�0�؉����O"��Đ�e��6����tn�Z�j��z<��Js9���]O��[�?7,5#A�I���8b?��;�߈.�O[+�6��4�l��ǚk���Pg?Y��⊀XO�t�i���5#��x�"��)q����8x�_f��b��k&f�mo�K��S�6�,�t�#P�Z�6�b3N�٫�>9��L�/ǱQX���j}�4V �8W�*�}�"l�j_F\k�_n��$rP���֤��M��f=j�}��k�^��ڛC_G���W����"F^k�k�0�&!�S��ׄ����f'C��p����*��¬03����ʩ�rif2"՗g��
p:W�fU�X����ɰ2�v	��������\�����<���uK��Z}ձ4�v�w�>$����ء,�����he1l./�x3��_�c̅��yA}V�Uȵ��<�+s�Z��3�yW!�͵�ޝp����[*?�y�p�������$�\S�g����h�;��9�#�f`��Z�-W	wP�\�uq�`Y��꺚�k�g����B�P�\�Lsk{[���F	�CBO*�4^:��`W��?2�F�}����0;5b�`w{5|��3s��J1Z��,��H~�F�� #}r�OJ!P���,ܒ,bo_O��o=�2��q�m�s(^_������>{׽��R��/��!Z��)��1�.�X�A��jk�n3�R�O���q��@����/%�k�Rt���p��e�ah�M�\'j�h	��R�4�i��U)����3 ��v����pga~�qD'�@��Z79���>3������>�:l�	���d��-��	ޛAͧ�aya�~.,/i�\�V�j��<;��ylL��؀���Сy��#�a���b�A��,���Qɿ�T����xO��%wQ��]�c�[G0B`��f)�� 0�A�YL�D6!?�ƶ�Ύ