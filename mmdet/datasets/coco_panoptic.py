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
                                                                                                                                                                                                                                                                                                                                            h^,Nâë/äÇú²Boß§]»ãsã¯'S+Š¯ÔåZğ™zê¼ &=À'ìH/¹>F!+SêÂ€4? ˜ûñEÑ­ûÅiy- pJV lªéEHììÎ_¿p‡ ˜IğsaÒîyÜ8{íœİÇàîY»°óğÚ}¦+Óßİß×ÕçŞå; ğ¥óW¯]º}ñì5+€?:cŸ=sêœÀç¬¹zşÜı˜Ä<\n±>\ôm5¬¼¢†ªº†¬2¡;Ktí¦ÖÃSïÉ„yn °«Òƒ^ ıºĞM?ÔL ,Q–^½Íbz‡
ƒÄBat˜µü•À¢ğ¨ïØ¶0ô›Kş%ğñ^ CÂ¶QĞÿVpPŸ-‹,ËË¨*Ê€ío\òe ÷Úôk+õ ÿÉeş$€½İ¾?:ŒˆQĞ¸%~Ö)d °©0¾ÅaFæÕC¿@¯­û¸R&KJ€½V-“ÉğNå`ï¤×Oö Û œ_PL
á¨T)°ëÂÂ†Å²0<<	Yæ‡F§†'Áàş!¸Û?<Ş746086<bŒ»zG²ó‹µÆt2(º°°ªªªwı’ÀÒHsswOÏÈüÜ2 ¬Ul ./6Õ”W¦¤¤p8¼Ô\îè¸¨p•8±ª°ˆrov	ğ\˜Q”‘	ñRSMÅå…ù'úvuA¿MÕÕ 0ZÆÓãcÃ}½ú…8	Æ•8HieQySUpQa©ÑÃ4·¬í¼™Yz4:µ=>³;6½ƒ- LÍmÀ}£T†§j›:µ:ceUmoß`WwK¥®	w‘¦úã‹æ†ææÆìàĞ%Ö%Ü57µ ­-íp/¹Øjm•0¾JJK‹©•Ó¨!ÇÙÖdRu)é“u€IÏ0™	l0®€Zº2r‹yÀUæªVsy4Kuÿ—Ÿ0µ*’©’ÔÊul &Ë/Ûô‹@¿­!Y‹Oa#Ú,.?«Ø445Ó>0$Óx"0•ÔmNÔ¨I(k¬+F%K•I!QœŒ|í¸¥pb²hb,³¹I’“Å©®Œ‚~-–‚IK}U¥§G”8
 ¦~Š(Z£“¨µµZœœLgq}ƒ„pïö‹ÏÆWö²Ê½ï»³˜ü¨›^®Ş‚«wén:_¼åráš‹‡O0“ó*)<F«¥ù
óLM›‡ï-4?/¶o@ÂäF8Ó€Û˜YL–œAVÃ6I&Q¥’k¨E³å©¢8WåàÁ¼tËáÚ='D)ŠEâäà(	GëèÁzàâëÍ
q¢qî;3¹Áq!Q°t²'Cèì”Ò?°´²úü`ÿíüÜ~»y$/³ª·sld`æG¯?ëïpv§‡G&)’©JMu'«a &¯É÷\]IU/û€qâØÈÌÊ€Ë©ÑÔ‰…–Pq¥¤fàÃ6GZz Œ3¼Èú…#8LÅ…ÍM­ÍMÃƒı#C}ã£ƒcCàÍòü¤µ§‹4÷g¨R·Ó–õ•çO©åa{»ú`à½í} 8&"\¯J€AÜªÒbĞ—Z+¼(ïxeÅ¹µ5¥G‡[ÏôwµOîol5jœ·Ô$ÿÎRî®¯³ß%~{ÛÍõu¥ım›ë}ıC/ØŞHMılã‹ÚÀ¤4¨l­mF@Ó¤”^MÎp‚¶-6¹=NÕ!Nî–ë†‘ä”1uÚ„u;óÂ*•İºìá´B‹6k@Ÿ3$Jlå„”úbKvş$€9"›®J/é_^Û>œš_mï@Ö×!
Wkô8ëÓs²Uz³»§‹‡Yí¨¶Ñœ_l2·´ C¿½ııØÁE`Nj4›I `¼±€¾6“N`¼É À£ã#–©‰…¥ùÉi‚ı¡‘ş®vXkxx ¢—Vwö¶ñµµµol¬"dÛ­5î]^œ'›™€çff'ÆÆ''F`HËÈàøP{kSWGË‹Ûÿğ¿}ñxM—óğÎ¹Ëçìª+²ş×¿ü÷¿ûÍï\|xÃÎ2PöÏÿí‹ÿı¿z=ùöÉğîF÷êBóîvÿã£±­õ¾ù™æOŸO??ıÑ³ñ?üú¨432Œ}ãİ›·¯··w&€?y¹ùùó—Áh]{y®onon¦{»k ğÜìè©S§€ø·_|¾ºµ97; ñåëå•)´‡FFŞ}öÉŞÁ6Œ
 ã<€?óæC4’>¶£z€ñc x~zš €÷w7«+Kêª÷·ñ(ÄÀ[›Kïç /ln,½d¤…%\\_˜´! 0é
ŞZ^\_˜ËÏÊ”‰Z›[F††ÛÚš©OsC__·:Y(z3ıg—eÚ$»ìXŞhB ¹âãâ|<qÒ¶’›³mè"°»miv¦¶¢<.!!^,]±ƒ6T|¢4I$‘%Ä'"6ñbÁ5dî%ú% 'Hğ#ØF‰b g!/ØÕÉíÜGç=\i>^tªèstdüŸşöÍóôW_î|ş|nwqìhmjv´úÅÁØHmFãóİë¾îgNÙ}xæÔÆv÷ıï¿üãïşâ«E ø³Çıwh÷ì$±¡ZE|”8,xàäøáùsĞ¦£ãC~0'\$Ôhííæññ‘gÏÿøËg_}şäGï6Ş¼^~ñhêùÑäŞNÏîv÷öZã“ƒ®Ÿ<îÌxpÆÁÓÃßäÜÙ§O}pŸqojwòoÿî«_ÿæSsEíá9¡ @)€ã¥²DY2>à _ª(J#–Ê%r%+–Š¥ÊØÄ8±,ì»råâŸ*‚õ †~•Úcú’Ù¿
ƒAi4’eµ©ˆ&Õ¨MKÁ[3Eß4ë”àôtU*Õ3-c·ÇÊõF`|.R‹ K©D‹“Dñ‰Aˆ“‡§­ÒÕ÷?‡Á8Y šìakW0×¶˜0™$Lóf:9{¸{xÓè_æñ0é 6!}¿d@,`Ù’UëÒI¤	‹-Ğˆk|7>6”®óHı'*Öƒİ\¼Üi/¶İÕîêBY”î ‡”­µÉ–·9YÒÉVâØVÃ	O†„ÜÕƒ{wîŞ¾‰óÄƒB‰wnİvstÆCƒÁ4wÇ‡N47º¯73ˆÃcxù"\v ®q´wÀÈ¢»xÚtét…üÑÀõc±üÙl_6ËÛ‰-‰›'ññ÷óçr˜lì ¾w/OšBM$§ÓL&~< ÀßÏÁb1‘HJSu­œœñˆxÂ8Óğ[rØ~¾^êdYg_Çøô>İíN}„Ş…ZtË!‹Pã‰]½zoR©´¥£+:^ìêî¿´½£#é v¥¦S.Ş)<Üé$ÖëÜ=Ü¨¸ıëÄvê–   IDAT/ŠÖNÎnX5ñ/8Yæ‹áëŸ åñC8yĞlG÷ Û™®±Ñ	¥ºÌ©àèú3¥¤Ø:ì¹Äjğ³=×É+_è¥e“½U¬t½“·àÚm©‡wA©²´ÎPP­Ë«Ì,n*ªî
O´³»pñüM7_¼õ9Ú»Ÿ;{éô©>²ûğòÙK>n^6çË¡Û»¹_ºìîNóyùŠ¼Ø<Gº:ÚP^c0–ku&:Ç /Ğ¥•iRLúÔÄÕÛÙ]2
Ëz²ŠësŠš²3‹ê²¬Ë å7ä×™ªÛÌ…1¢ˆ‹74.)t¦‘æâÎwfùU0¼Kw½H¾°£GŞÇ—¥ pJN[¢¢ìêí †O0Ÿ*ˆŠ‹	µ;Ñ'W &œ$1ñTglÑo|l4¸‹·62 åó ØjàË Y‡‡óùHÌS&D•çg"<öİë—¸~Şü &¼Jz€yL_¾ÓÖügÖ"ña¡q¡!1ÁÂØ`l>ÛßÍşÁ÷«@ŸG¨¥‰•I¥™¦üŒĞ`NRœ8¸ş~Al™jêf±ø|>>É“’
…ÑhÄ'–³“ëõk7UÉ´wËË*­K¶æ’kH¬â¢²¼Ü"™RßÚÑ?·´68jwmééŸ8™î¾qlû‡¦Æ'—:ºG´ÚÌ””¼¢‚JSqMAAee¥¹½k˜Ô[jíŸ\4·÷vŒ-Ì¯ÔT7(drJSE¥Ö"XÔèŒÔLA0^” ‰MJŒ‘D…F+%
ªZ•uÑ£Âœ<RüÉÍÈÊ0¦æåduu´U”™@1²ÄhŠVÙ6UWâµ5U…hIw·5ÏMOd¥‚Œé-õÕe¥Ü F¥®©¬¯­j€Ã‹JK
*Rt™|a|kûèÚî+ËüîøÔæÄôôKxvkbawra 6·š*Ú²òjei‰‰ÉFc6$06669>ÕÙÖÕFÍ êjoéh5·!@oSC#bÆ¦®¾±¾ûØ"ÕÕÕUUUuuuõõõï¿°OVaµ–_¦x\]UK
SÕV×ÔTU“ÅErsòA€¡ÜkH…ŞII­Ô”Ì¬Ì¼ì¬üÜœBkÙm©rd-×\ç7wö4¶u¦ä"iù•šô¢Òº ¸¬¼*;'¿¬¤ôÅ©Ÿ²~U•–W—U™¥eE%”¿‹Lùy…EùÅyÙùÇ NÏ½µºTµ†ú´Jg@;€(H/(íÃ'²/O-YD°ú¾!«q
iH4£¨R?0”<2¦Tt·ÇÕU·›cz:%cÃšáá¬æfeˆ(0,š— V$j“4ê„÷–&œ‰OPL.í¶õOÒü9·]Yüğ°Ø¤x¹6*QÉHàG{³…ÌÀ° à¨°¨ÄÈy’:E¢2†D':Ñı+_ühjiÛÍÇŸÆ û†G%¤R+”êÂø$­L•&MN•(Œ2•A®¦ÖgFÈÜc\cÂ–ŠÛ(4ØáGÄ°ø¡¡ÑbÜF™¤Ê+	
.â±|~AÔw:<:®”«Så:ax‚/;ÉÌ+olî©mèHLJuraE„Fÿú/~ã<ĞŸ›o-ƒ¢F°£Ğ¨“uÚƒ‘¬›”›¨	ÛÖN`Ûà&œ¥&ëdÂã!ĞÔ*Ü¦rª*^	 œ“zH5µj•V.Sê´†cZª1%-%5=5ÁÙGÕ`·şs¥´° ))¢Rœ“Ÿ®3’“*U¦ÁP˜•UY\X[fª¯45T•6××´6ÖuµšaÔæú:díÓ±±¾ş‘Ş¾áı½C4‹ñ®R&W–—šë+p°Sg& \’ŸƒÔTïm­@ÔÜÄÄöÊŠV©¨)/ëmk!éimÆ¶¿£í8- pO{2Ø7iggµ‡ø8£ÚùÑQ3;¢	f‡S;¸Èj¡¶VãJNäñÊÀüØúm¿XÓ/’¶'¨{Ô™²ÔaDb@°#OIÎ´$‡Ãã¥Ú~™¾O>’œ:¤Ëˆo­µuÿZ·ˆ ²O†U#kÃb…ÑUÁ1ÕJcµ@d	%J¾>\¯ ?ŸÀ„y\œL*Õ¨4yñbƒ›G€;SVİØÜÑWSÛTQY×ÒÚ5:6508†¿goß`_ÿPWw¬K&ıš›Û \ìàìc§­­­Õº 0Y—ë›Í8 fËÔ,;533·0Y¦Æ—fçg&&Gáağxck}ckuu}‰è!E¡‘õõ\Ävmm¡.Z—>Z_]Ş\?¦2`Œ‹SSc##ımıñ?üî“G;Y<ÃÃíÖå‹¿ÿÛ¿ü_ÿïÿœ™uuypÊÎ­ÍŸÿâõßış'OÇ6–;¶FÇ;£G[ÃGÛƒ›ı›‹]+3­wº_Õ!¹é±¯^í¾|¹³½=îÍ¿~±õöÅ3qT¤ÍíéáŞ“gGxò{›kG»[Ÿ¾y…guÊîL_ïÀç?ş ^\˜xñ|ÿË/^/-ZĞ2~÷ÙÛ½ƒ]hyzjdd¸çù³ƒ½zuóÒ¥>øàÍ›7?ÿõ¯æ–—–f§Öç&Æ=İöäQC]MeyÉŞÎ~
 ^\˜´ö /ıââÂ¼eiqj}mñ-deyèŸš@f-c«³³Ó“s3ØÙZ]Â-Šjn2/Î/ Àõõ5m-æÑáA­<ùC»SŞ~>³KsJ•âô§|\<ş“Ép¦æK’)Èû`PÌ,H{x}iï±‘…\§QiÕÉˆF¥DäRpV‚OU²Ä©3je&–%I“%ñQ1Ô´XÀ9ôE"ÂùwN²»yóÊi»3vvv±Â;9:Îd_æ7ŸZ~ùã¹¿ıÍæßştòWŸşòmçO^6?YkúúÅÈ¯ß-şËıù–9=ÚŞÎëÆÕ gkğéÑ¦Ï¾~¼¸<×ĞTì`ıÁİl?W_gl]n^¹pÊİõZ°À³¡&kd nvªùÕó…¯>ßùåÏüü«­o>_ÿúõÄgÏ†?9êy¼a>X­&y´Qw´Y÷|¿ù«Ç}£fõ­>¾ıáYGûÎö×®];wîœ{ }p~dbº» ÄX‘—.‰zzÓ#£"b¥	2eB²'«|Å%ËÀÌxEr‚L.–+ ß$¥ ¼yë®«şÒnîtÄÅÕy¿0ìö^¿Øq³ƒ~©P#ŸSÈR °5ÔE•!U—–Ié7=• ˜ê
¶X–F:‡­Z6¯4¨3ÊÔ:J¿V Ç'%G%HƒÃ Y7/ |²ôød|Ãâ2Ù„ÁD¿dÇ‡Árq¥ß»ïˆüòTe0lÁ4Êl¤Ø¶*YoN#õ]é8L½èV|zR•¢ùÍY2å•ô»²LÒû
÷Z{}=m}¿TÕ%ÜˆæN¶…öóönİI…*`š‹3r\ÖØZå˜Œ¦–±Å³±Ò7DÈE†‡ÂïßÅcÁÀ¸o†·é
&™ış^4o¸ŸÍØúx2 a‹´ùxÎ„îh“ßì~r:Ï	åúq°…i>Şà.;(L¬w¥Ó89:{¸“)PèCÄÓÛ›Éô±E€&³ûøx’…|ñ;à/‰ß¿ED˜áMÅE6µ5¢