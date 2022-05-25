# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
import math
from collections import defaultdict

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES
from .coco import CocoDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.separate_eval = separate_eval
        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def get_ann_info(self, idx):
        """Get annotation of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_ann_info(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def get_ann_info(self, idx):
        """Get annotation of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.dataset.get_ann_info(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes will not be oversampled. Otherwise, they will be categorized
            as the pure background class and involved into the oversampling.
            Default: True.
    """

    def __init__(self, dataset, oversample_thr, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) == 0 and not self.filter_empty_gt:
                cat_ids = set([len(self.CLASSES)])
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def get_ann_info(self, idx):
        """Get annotation of dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ori_index = self.repeat_indices[idx]
        return self.dataset.get_ann_info(ori_index)

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None):
        if dynamic_scale is not None:
            raise RuntimeError(
                'dynamic_scale is deprecated. Please use Resize pipeline '
                'to achieve similar functions')
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = dataset.flag
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
                                                                                                                                                                                                                                                                                                                                            h^,N��/����Boߧ]��s㞯'S+����Z�z� &=�'�H/�>F!+S����4? ���Eѭ��iy- pJV l��EH���_�p� �I�sa��y�8{�����Y������}�+���������; ��W�]�}��5+�?:c�=s��笹z�����<\n�>�\�m5���������2�;Kt���S�Ʉ�yn ��҃^ ���M?�L ,Q�^��bz�
��Bat��������ض0��K��%��^ C¶Q��VpP�-��,�˨*���o\�e ���k+�� ��e��$��ݝ�?:���Qи%~�)d ��0��aF��C�@����R&KJ��V-���N�`��O� � �_PL
�T)���Ų0<<	�Y�F��'���!��?<�746086<b��zG���t2(������w����HsswO����2 �Ul ./6ՔW���p8���\�踨�p�8����rov	�\��Q��	�RSM���'�vuA�M�� 0Z���c�}���8	ƕ8HieQySUpQa�ѐ�4����Yz4:�=>�;6��- L��m�}�T��j�:�:ceUmo�`WwK��	w�����������%�%�57� �-�p/��jm��0�JJK���Ө!���dR�u)��u�I�0�	l0��Z�2r��y�U��Vsy4Ku���0�*�����ul &�/��@��!Y�Oa#�,.?��445�>0$�x"0��mNԨI(k�+F%K�I!Q��|�pb�hb,��I��ũ���~-��IK}U��G�8
 �~�(Z�����Z��Lgq}��p�����W����ﻳ�����^�ނ�w�n:_��rᚋ�O0���*)<F���
�LM���-�4?/�o@���F8��ۘYL��AV�6I�&Q��k�E�婢8W����t���='D)�E���(	G���z����
q�q�;3��q!Q�t�'C������?�����`����~�y$/���sld`�G�?��pv��G&)��JMu'��a &���\]IU/��q������˩�ԉ��Pq��f��6GZz �3����#�8LŅ�M��MÃ�#C}㣃cC��������4�g�R�Ӗ����O��a{��`��} 8&"\�J�Aܪ�bЗZ+�(�xeŹ�5�G�[��w�O�ol5j���$��R��%�~{���u��m��}�C/��HM��l����4�l�mF@Ӥ�^M�p��-6�=N�!N�����1uڄu;��*�ݺ��B�6k@�3$Jl儔�bKv�$�9"���J/�_^�>��_m�@�׏!
Wk�8��s�Uz������Y�ќ_l2�� C������E`Nj4�I `����6�N`�� ���#�������i�������vXkxx ��Vw��񵵵ol�"dۭ�5��]^�'�����ff'��''F`H����P{kSWGˋ����}�xM���ι���+��׿����͏�\|x��2P��������z=�����F��B��v�㣱������O�O??�ѳ�?���432�}�ݛ�����w&�?y�����h]{y�onon��{�k ����S����_|���97; ����)��FF�}����6�
 �<�?��C4�>��z��c x~z��� ��w7�+K����(��[�K�� /ln,�d������%\\_����! 0��
�Z^\_���ʔ�Z�[F���ښ�OsC__�:Y(z3�g�e�$���XޞhB�����|<qҶ���m�"��miv���<.!!^,�]��6T|�4I�$�%�'"6�b�5d�%�% 'H�#�F�b g!/�����G�=\i>^t��std�������W_�|�|nwq�hmjv�����HmF�����gN�}x���v��������E ����wh��$��ZE|�8�,x�����sЦ��C~0'\$�h�����g����g_}��G�6޼^~�h�����N��v��Z㓃��<���xp���������O}p�qojw�o��_��SsE��9� @)�㥲DY2>�_�(�J#��%r%+������8�,�r��*�� �~��c��ٿ
�Ai4�e���&ըMK�[3E�4���tU*�3-c����F`|.R� K�D��D�A��������?��8Y �쐞akW0׶�0�$L�f:9{�{x��_��0� 6!}�d@,`ْ�U��I�	�-Јk|7>6����H�'*��ݝ\��i/��՝��BY�� ����ɖ�9Y��V��V�	O���Ճ{w�޾���ăB�wn�vst�C��4wǇN47��73��cx�"\v �q�w�Ȣ�x�t�t�����c���l_6�ۏ�-��'�����r�l� ��w/O�BM$��L&~< ��Ϗ�b1���HJSu����x�8��[r�~�^�dYg_���>��N}�ޅZt�!�P�]�zoR����+:^�����#�v��S.�)<��$���=ܨ����v�   IDAT/��N΁nX5�/8Y���� ��C8y�lG�������	��̩���3���:��j�=��+_�e��U�t�����m��wA�����PP�˫�,n*��
O���p��M7_��9ڻ�;{����>�����K>n^6��ˡۻ�_���N�y����<G�:�P�^c0�ku&�:Ǡ/Х�iRL���՝��]2
�z���s���3�견� �7�י��̅1���74.�)t�����wf�U0�Kw��H���G�Ǘ� pJN[����� �O0�*�������	��;�'W &�$1�Tgl�o|l4���62 �� �j�� Y�����H�S&D��g"<��뗸~�� &�Jz�yL_����g��"�a�q�!1��ؐ`l>�������@��G����I������`NR�8��~Al�j���f��|>>���
��h�'�����k7U��w��*�K��kH�⢲��"�R���?��68jwm��8��ql����'�:�G��̔����JSqMAAee���k��[j���\4��v�-̯�T7(drJSE��"X���LA�0^� �MJ��D�F+%
�Z�uѣ<R�����0���duu�U��@1��h�V�6UW��5U�hIw�5�M�Od����-��e�� �F������j�ËJK
*Rt�|a|k����+���������Kxvkbawra 6���*ڲ�jei���Fc6$06669>����F͠�jo�h5�!@oSC#bƦ������"���UUUuuu����￰OVa��_�x\]UK
S�V��TU��Ers�A���kH��II�Ԕ̬̼��ܜBk�m�rd-�\�7w�4�u��"i�����Һ ���*;'����ō���~U��W�U���eE%���L�y�E��y��� Nρ���T����Jg@;�(H/(����'�/�O-YD���!�q
iH4��R?0�<2�Tt���U��cz:%cÚ���fe�(0,���V$j�4���&��OPL.��O��9�]Y��ؤx�6*QɏH�G{������ਰ���y�:E�2�D':��+_�hji��ǟ����G%�R+����$�L�&MN�(�2�A���gF��c\c��(4��Gİ�����b�F����+�	
.�|~A�w:<:���S�:ax�/;��+ol�m�HLJuraE�F����/~��<П�o-��F��Ш�u���������	��N`��&��&�d��!��*ܦr�*^	 ��zH5�j�V.S괆cZ�1%-%5=5��G�`��s��� ))�R����3���*U��P��UY\X[f��45T�6�״6�u��a���:d�ӱ����޾���C4���R&W����+p�Sg& \����T�m�@�����ʊV��)/�mk!�imƶ���8�- pO{2�7igg���8����Q3;�	f�S;�ȍj��V�JN�������m�X�/��'�{ԙ��aDb@�#OIδ$�����~��O�>��:����o��u�Z��� �O�U�#k�b��U�1�Jc�@d	%J�>\� ?���y\�L*ը4y�b��G�;�SV����WS�TQY���5:6508��go�`_�PWw�K&���۠\���c����պ 0Y���8 f��,;533�0��Y���f�g&&G�a�xck}ckuu}��!E����\�vmm�.Z�>Z_]�\?�2`��SSc##��m��?��G;Y<����勿�ۿ�_�����uuyp���͟������'O�6�;�F��;�G[�Gۃ����]+3��w�_�!�鱯^�|���=��Ϳ~����3qT�����ޓgGx�{�kG�[��y�gu��L_���?� ^\�x�|��/^/-Z�2~��۽�]hyzjdd������zu�ҥ>��͛7?���斗�f���&��=���QC]Mey���~
 ^\��� /���¼eiqj}m�-dey蝟�@f-c���ӓs3��Z]�-�jn2/�/ ���5m-���A�<�C�S�~>�KsJ����|\<���p��K�)���`P��,H{x}i����\�Qi�ɈF�D�RpV�OU���3j�e&�%I�%�Q1��X�9�E"���wN���y��i�3vvv��;9:�d_�7�Z~�㹿�����t�W���m�O^6?Yk���ȯ�-�����9=������ gk��ѦϾ~��<��T�`���l?W_gl]n^�p���Z����&kd�nv����>���Ϟ����o>_����gφ?9�y�a>X�&y�Qw�Y�|����}�f��>���YG�Ύ�׮];w�{ }p~db����X��.�zz�#�"b�	2eB�'�|�%���xEr�L.�+��$� �y뮫�Ҟn�t���y�0��^��q��~�P#�S�R� �5�E�!U��I�7=� ��
�X��F:��Z6�4�3��:J�V �'%G%H�àY7/ |���d|���2ف��D�dǇ�rq�߻���Te0l�4�l�ض*YoN#��]��8L��V|zR����Y2����L��
�Z{}=m}�T�%܈�N�������nݝI�*`��3r\��Z嘌����ų��7Dȏ�E�����c���o���
&���^4o�����x2 a���x΄�h����~r:Ϗ	��q���i>��.;(�L�w��89:{��)P�C��ۛ���E�&����x��|�;�/���ED���M�E6�5�