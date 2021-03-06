# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import (DistributedGroupSampler, DistributedSampler,
                       GroupSampler, InfiniteBatchSampler,
                       InfiniteGroupBatchSampler)

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset, MultiImageMixDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(
                dataset,
                batch_size,
                world_size,
                rank,
                seed=seed,
                shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset,
                                   samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if (TORCH_VERSION != 'parrots'
            and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     �,���FN�%O�ǭ��'�PM�c�y� �1Gz)��O���2�Ag�TQ<�\�Z�xʕzҙ<a	:U�|���Վ;Y�uʍ]k���NV_Yji������ߺq ��g��81gj� �zq@�!�P� ����ñI!1���9�����ϕ�0>!,3: *� ���y�����Ӓ�H�{�Ug�U���f�7N�:���P%��������̑Z�f�a��ZFy���s�~Ȇ{А�eF?���S*8X��[iK��4�5Y�[O˓Ԧ�ߪ	�)� {�k<[�͞NVD<;@���u�,(���:���Ѫ�gJT�u\iI*
a�?R$G�ys�խ���Нs-/_���u�[݆�� ��@�?�}��|���{o��~�5�9�uw�_^��צ�b���.�kF���n1��%h*WU����^�1Qͥ��*vf��|�J>|���L�G!�o��A�?�����^�Xo%J�N����(?�3����1�����%����VkD��X��إ8dd�)P<�V2X�W��	1��Ε���ͲhH��ON}<)tor���A����J�����E�D�U>��{s�/�F�G.C3��r��=�V�⎟.�3E6+�Z�O��A���:����4������c�7�}���o��FJ�=��L�Gg�o-����^�|����uw��U���3�	;�� ?{2����fۏ� �wN��h:4R�0 �j{����/����� _��J} +/�u0�#�"�&�<��~y�3@☬,h��T��xF��	���*5��F�Yi:r8�G�d�()�@���^�=�}���\���=��!xC�*�J��[ #�ݧ0�����.�S��'��Og
S�$:7�-���S��lI�P	�D��e��5UpoG� ���D`A�x&���i	찬l���p<�@\t��`�M�@�)��Tj@*9(���) �蛄�Fe�OG:��t$!�hb�T̡��}1I�c�ƥB�cҡ�q����c��ݎ'`|��i��
8À}��'�%b��p�0�F�AeA�	����Si��t��'�m�
!��}�&A�D�K�4C�R�,`c���m�y%�y��rw~E��]e��*lPm�$\X�SRb*�p�՗5��w�vt�v�6�9]���|WySuS_S�B�����E��s�V�k��)n.i*��ܿrT��KZն�W}^Y���T6.�,m)i�-\����>�idAx�l�A{%�x�\������jG���3��O�j�d��j�*{�W�:�!d8�v:���'(	B;�i"�����9�Ynu+r+5�jEn�2�\f)a��<]>[ib� F�HC���b�����$Ӳ�֝t�4�	"*�d����`���BP���&��BYpS2Řΰb�Pz�"
	WL�8-Ba��Di&M��V�yJ[�iQXzT:1.���%d�1�$,#�� ��dn&��E���B��"��Öʖ��%��1�#'����Ȕ`�B<S�T�!�B�D�*�b9&e��l�����}��(֛<Wo���\C���*��x6 ��Pʔ*�"lby�A�]���X'�N�V(ZW�D�Jd104�1G���8��jR���Y�8ǢɅt�&�"���.��yr%�U�)<.M$48]5�R����At ��Q��:0�/P�a�A�0��MUm�\k���K���A�0j�:�:Gc4�M��<�Ң���l������<�dk��`�|�s��
ֽ���j��ݎ���鮂������j[U���f,q喸��&�I��g��7[�LR���c��F�*תw�5NKE9Xw׽00���0���γ����548��]��M���������������;��V�[`.��-�/��h�j)���B�����މ���S����஻y��N�:�n��H\\��(mqV�k��ҶҺ���᪖��V��T��J����ԙ��Ut�h�[��ڶٲ�Q�����m�k`��Q�s���%J�K�w�Ŧ��ʦBw# Xg�׉�������p{��������N���lU	�����_<1�k�7|���~��Ə~���/΍��z��W.�}��_���*r�����Ꭱ���Ѻ����W��-�m��;z���c��'�gg'���*�r%�lH��)��׳_������J�;o����{o���?��̻o�{?|������p񏟮�᳓_~q�����<���^��ֽ��؇�����?Y���������k5���^}��;w�uܺҲ2��ʳ���Ǖo����\�y���Y|�\0�s�[��}�z.]���J��|��l��T��X{w���(�r|�gj�yelemba��������≾�ɚ�����^��~ct����նk�;o�����WGe��m��F���Oz�/�<����텩��bkUY�� �fg�;����n��޾5�~�ep�61^��0X��1��Ҷ0ױ8߹���   IDAT|bpq��ĉ�����CЩ�����SK=gN��?��Yi;��~�T�ƙ��g;Οn;w�y}������|{`�k`�/�p����˕KK�����=�������������-p�ٍ�W~9��?m���@�����΃�?��20���]���П�|��|��6Yt��}{V������y83"T�͒cм���}��>��\�43:1Խ4>��Ro�	�b���\����s2�H��BL
�0{d?6"9
ڳ����h�Ң#a��i�;�EVw!|���D��z�D@���~�p�?�)���;��n�����A`ם<+���{�]�����븸μ��4M5mӴq7��I ��;�����03�����H F\��iR�v���]�]�>���ݽ��<���}}^�^�BIw�{�s�_	�?��/�G��]�v#������mL?�Wt������ȕ���Њ���)k�9[�8'j$	>u�[�.->���l�E:i�ف�V��}�F�6�K�w<]�ֆJmc�􊒻$1���6��z|p;�`�H��6g3�`