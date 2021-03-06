# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv.cnn import VGG
from mmcv.runner.hooks import HOOKS, Hook

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.models.dense_heads import GARPNHead, RPNHead
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='ImageToTensor', keys=['img']),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(
        ...        type='MultiScaleFlipAug',
        ...        img_scale=(1333, 800),
        ...        flip=False,
        ...        transforms=[
        ...            dict(type='Resize', keep_ratio=True),
        ...            dict(type='RandomFlip'),
        ...            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        ...            dict(type='Pad', size_divisor=32),
        ...            dict(type='DefaultFormatBundle'),
        ...            dict(type='Collect', keys=['img']),
        ...        ])
        ...    ]
        >>> assert expected_pipelines == replace_ImageToTensor(pipelines)
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = PIPELINES.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg


@HOOKS.register_module()
class NumClassCheckHook(Hook):

    def _check_head(self, runner):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        model = runner.model
        dataset = runner.data_loader.dataset
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert type(dataset.CLASSES) is not str, \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'CLASSES = ({dataset.CLASSES},)')
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not isinstance(
                        module, (RPNHead, VGG, FusedSemanticHead, GARPNHead)):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)

    def before_val_epoch(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^ԭ����N]�W��j�`���t�s��!�|笫y�X��ݺ*O*�4e��R��lP_2f��q�Ϲ��E6E���b�]�fU�-r�I��kh��n�$�,����	�r�ؘj0ץ�#t�G�@��N5�ѕH$����S%��(]k��x�;w{���
��*֦j��
~y�t�/k�/oi���Y�ޖ��y) y�r�m�_�% C�`��� ��%���^�Û���]ԇ��U���o=t<�k}xӆ<��zr���ݡ���߾3�֭���F߾=�������ׯ:]s���~t���5������r���yxӉɭ���=��=��e������M{۝[�-�η]ߓ�wd7�ŷ\�m���|m��@������wV+/ ����{��ۋ����f��NW����C�n����e���K��7��2��S����e��l��m�k���ژagP> �WD�v����ܝ�����VAۛ����a���6i$cn���J�>C�y�idΑ�;�0i��4����M�D��"p�g�o�F���5��YZb4�6�%Q��0yM���$����n��n��'k���Ƴ��wV8�L]/97_�o���F�wq�r�Y�.�3�?���g|sb�Q�.u�,����im���̠3���xc�va�df�)���W�-T��g�ʖ�<���傾D\��Y�<Q� x� �21�M'�b�S���/Ћ�*�{�(��T�ך|@\3SԥqZx�$f����}��leت�gcV3��1$bt����	w��h��H1 <֗�,����y#=�~'z��sxոi-:g)��(0�tLz�Us���R_��@�n�f_�yw��Dù��š��ޒ���	W�X/�)[ۙ`�
\�~{��V�XL5fc����hi��q�t�woaq�����Ѭ�vuw�t&����w۝.����hR���&ǭ�W�E�����8�0R�����w�B���8D#
>��U��j�G�M���W�vV��*5��+O`T���T�s�B��/���m��k��i-;%-��f3�~�9����,-�{(�% 7�7���ٳ�� $a����u�g<���<W�eS��K����X��G�m����2��!��`DR�\�R��+'Ȅ�s�Ӄ��5���
7t��=f��@���T`�e��������8Im����)��&H*b��ј��$�V�Kk�����U�!�Rh0�Pu���B�b.*[3��H#�!I�*C��A�0X\�����Ֆ��v6�ý-I`�0&��1�9��6���/����v}�^^�!�{s�]o�$�hʖ��-y���P}��vQFuQ4>���8�ׁ� � y��� X^jP���e&E���Ԫ�0��
���(/��V�;�%�qU9!7v�����gx�㷖{���V���;�,;;Ysu���po�vc����sn>yu*��rק޼Y������#�O?P�sGx�\�=�?7��᮸�ׇ��O�7'�#g�a�=!�������I��,��L
�;��\��@�/��eQ�5�`����O�����g?����~o�E��Q�jw���i��`/"�N�s'L9Ö�!s��;K'Kзg�M*����n�w�Gίmm�llLnl�\��<����1{vc��������ّ����)��qň��ϐ2d���|oт�d�Q��S�Lkrƕ[��w��?�=���&~qg�7��O��-{բ6A���ȣ��ۓ������0�.��
�`���u��t5����ژ��� ��Cp]aHmAp$�`8��f��y��i_7-ů/��+�`�.+��(&	{���ៗ� g%��~�>���
0�n~�T��"��~S�O��[`v!4]y~���Cv3����"�]��J_z��@��b�����KM�h$ {� �$q���ѾIP����qo=8���`�����;�u���(FK��O��������G�NP7#:Ն�pcJ�����x�9����M��A��% C���d�l�on�/�T���t&;їn�~��
v�S�i�4�\�u�
}��@#���(����N�?�E��.�> ��u�8�4�a��¸ƺ��Ҡ��S��>L�젪�hQC��&��:��,);':=#c �GZcj��y�<3t��X_w��`�I�a�6��;!/scNg�s��"���K�3�ag^��N��w� �t=��	��!�^vǯ�~�,�s
�����N�=l��� �L<M�������������}x�7B��!4����r܅~a`�omeTUY~���w&s.��t3Y�aǢ#N�pN"T?����u�Pn/O�!�C<ͷ��M��3Ol�_d�鈈�aaǑ�����ǐ�A�00 l�J� �&C���b���nk��=}.��W�ĥA��.�a:�k]NC�]o�h��.����w>y����oXmZ�N��>،@���� l�tP+,�Ij4��:X����M]��݉ߠ�z��<��1㔎[�CݵH���OSѯ�X6�-�W��*&�� �f�Qۯɷ��``�<1���l�{��-u��rY�}1��!S��%ݦ��~�l�}:\L�2�a<4�s�D����P��H)Hs��ڎ��\��sҝ��}�[̉G{��0�������_��B&#/���߻�j ��u����0���u�)b   IDAT�<�w@�w.ٞ�zߺ9������F�� ��G7�n�Y@�70��f���t&�[�7Z����/*.�v��/��.����ם[(A `dwɓ������B��lL�ю_����ȹ�tdř����y���1��C�0��'�mA\���ƽl��XE�p��9
 &�v7�����nC�{,����K+�vc�ѬzIB66����_K[< lG�ݙu�ax�V�{m��ٿ_��/w��*�ɹ��1W��l |n�agY��$�-�ʛ"�]�����f&$RQ��+k�Q���Q4�����u�+���������\�ve�^�/^�)��b���c<� S��Ep?m�}I��g��/BE`o�"T�e@{Xŵ���$�0���~��1���/xa�lz�pؙ9`O�$�N��:�Y S���S���fm�׵���5C᪾`�Q~�U��_��Ԭ��ι�`��h_Y�%w�]�v@�}�*���*���`����N�qjz�=��.kQ��==V{���t�-Z�J�t��A��n��3����x X!��5�t4��aL$AL�At�� L{}��/}s*j��էAߎ���ǡ_DV}JQ� /�K������������(���_��5�5i'ZKB`QA �HKB�Ł�� o 7��{1������g�0�Y�Ru�+�,z��"uE�%�J�å��/�v����L̥�˄^"�ᙺ�[ꣁ^܉����Dfp[�E�ף+U����x��mu\����R���V�H �zWC)/��Z2ԭ��-Պ=f�`�C�cd�6S�d���)UՖI���/uNfL#EX��+bA�ne��=����02���r3��	�m��vQ#�Aci|qV��o��mM�*x��
0*=�Pɗ����/�\��P�m�H��N)1*�:%�]�EFe~q��UL���A_F�W `|�e������ߏ��7�ޟ�.쬏Ե���Gl���	���f�eB��+� ��8��jy2ne<vm2��t⣫U.�_9��>�1�"K}LVb �+�G��w}"��\��b�ެ��;��5)؛LG�߽��e{��%lF]�O�?�}�������������m�*�t9�m�)cΜ�� <�S4m���X��o�M��ԕ�����no���g�77�'�{w.,]�~~kg ^Y__���ͭͻ�'�SN�#a>��p�U
�8J�yC��* �7�@���X�=Z�<��\rI׌%,����G:R���ý�3��!wg�{<�*Y�{�($A v, �hD%zWQT/To�l5�r�3����Sҳ��M�<w��~�y�#H߻�<��}�sp@� ��_��7L�8���V�1Hd3	���h_���`yw	�MFp��3����=&]��
pg!I~fFL��w��;��_ʄ���*��I0��� 3�m�$��0Ʌ~���#�K�0	��
p���V߽͓0��B}�u���iV'g�&-�I%0�EG��
pv�l	W<D����g��2%��E��S_ĦR�i&���6�c�(��J6�_�uދ����77�!V�.��<- n�}��th@�[�6n�7��

����"^.�.h�!̭{_ḐM��"Oy�� ��De��P]�+������?� �8?�h=�->8$) �K���Z���[���n��T�+d���&�`��P����ʂPY�0�l�7pQ�^��Ң}U�Hg����@Uپ�C�M���AI���yH�����"qKy���8[w�M8��K$�� 3&Lj�����r�?$�Ob���F�%�/����?*.�/���B��[���`�tt�|�2�� �0]��2��j�_� ����=���Ⓙ ��	��Q����<��pe����P_PT�FQ���I�t���!VS�� a����L�RIc*a�Hے)kf�;9�KOz(2��d4�I&Ӊx"����8p��S���ݻ{�����f/^<��_��/���_��_ll�&S���0X�g��D���k8`�UA�&��p*�vy�1"���s(�-�m�x,c95i9�6m�t`-8�P.z�]��E��JH��Rm$Gi�g}=� �-M���C��1Շͬ��MC���-���� ic�M�g=�4-��'Ԡ�Y_ä��v���&�u$�:0Qa��O�߹l�wݺ��sr��Ԛ�����Y���/���WϏ��U߼4J�d�i�|�������ܦM���	�u�z��.��ǃ����=���N��y�3���G>����	�{b]I?��~�d�+�Ϯ���ܲ��q�u��h7t�����9ǣ;�{7��緯yn_�ߺl�y�zcw���u�t����־l���ǹtBqqg����3��'���'䗏K	��%��Q�{i��=ݐ�/��)D}i./t\�oߝ_�m�8�
�=3��&��fxN?>��0#ZrW�z����ˎ�kŜ��)��S��������D��XqHUӕ�����ե �*$Lj,�3�ံPͱ����2���C��_�:G� �QSN��Jș	y�q�`��8,G�g�����3+}�~�U�?h.�;�VR�����2�XH�m/�|�����f�_��ZP����5��8�^K����ݓ��e��b��|+��Զ�l�n�w��.wץ�K3m��f&��\�����B_��
0��$ƻ�Y��6��p���l@�gE=��~îZ��$�
6�%��S�����V0��H'v@� <��	�rD���п�T�j�c-Fe����_���9�4rrAub~�̍�L��� XHb�K�fSCSI�$��p<>�(�Q����R��Ս����a	'"�I�W��B�Thr*��:֖S'�'�A�pO�ܠ,�ik���خ)'�������S]��l��ć� .���xO��h�P�{�d�@�l�0��A���_���C�����_��Qc����G�����GUm��1��Tt��i;Kƻ��]%#�a�ѱ����(�v�{�5�D��u7Dw��p�68��B2���3]��p`Z��=̕،��˰��I���!��QYe�����U��q��^s�US;>\
�r�H�^Q�+�͸xB�������2�~��uwy�,<�<c��%m�H�������3�=��J9���L0���N��ae�U 'a�D�I��M#�
zG9@?�H��>��^{LZcP4�l�1�`�.���*�M� 07�rX3���]k�����c�uh;���*KAk��k�r�{�:މ>x���Lj�=:�MG�K���(.���މ���2^<$�����Hc�~�M�)���a��z����W����?}����鵌ܥc��쐵!l�MԮ����:��O.��^m�v���:��b��<��r��%��t��b��U�m��T�Z�|c�rk�rc
�rl�3U[ӵ۳��f�6f�O�pv�����'�.�v�ؒ����)��*���{]qe��d�#�*_46�_3��\�'�c��.Ŀ8z�庖;��]u���/�N�Q���>��#��+�ތ�����_�q�߲�Z�/g����+���΢��fpkζ:iXJ��cJ��μ������Z\���ƥ0j|�)�ɴ�tBr6�>!����D{�