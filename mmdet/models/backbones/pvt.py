# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNetV1d


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(BaseModule):
    """Split-Attention Conv2d in ResNeSt.

    Args:
        in_channels (int): Number of channels in the input feature map.
        channels (int): Number of intermediate channels.
        kernel_size (int | tuple[int]): Size of the convolution kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
        dilation (int | tuple[int]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to
            output channels.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels. Default: 4.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        dcn (dict): Config dict for DCN. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 radix=2,
                 reduction_factor=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 init_cfg=None):
        super(SplitAttentionConv2d, self).__init__(init_cfg)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        self.dcn = dcn
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if self.with_dcn and not fallback_on_stride:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            conv_cfg = dcn
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            channels * radix,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=False)
        # To be consistent with original implementation, starting from 0
        self.norm0_name, norm0 = build_norm_layer(
            norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(
            None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(
            None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        """nn.Module: the normalization layer named "norm0" """
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.norm1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeSt.

    Args:
        inplane (int): Input planes of this block.
        planes (int): Middle planes of this block.
        groups (int): Groups of conv2.
        base_width (int): Base of width in terms of base channels. Default: 4.
        base_channels (int): Base of channels for calculating width.
            Default: 64.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels in
            SplitAttentionConv2d. Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        kwargs (dict): Key word arguments for base class.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        """Bottleneck block for ResNeSt."""
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.avg_down_stride = avg_down_stride and self.conv2_stride > 1

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.with_modulated_dcn = False
        self.conv2 = SplitAttentionConv2d(
            width,
            width,
            kernel_size=3,
            stride=1 if self.avg_down_stride else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=self.dcn)
        delattr(self, self.norm2_name)

        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)

            if self.avg_down_stride:
                out = self.avd_layer(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNeSt(ResNetV1d):
    """ResNeSt backbone.

    Args:
        groups (int): Number of groups of Bottleneck. Default: 1
        base_width (int): Base width of Bottleneck. Default: 4
        radix (int): Radix of SplitAttentionConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels in
            SplitAttentionConv2d. Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        kwargs (dict): Keyword arguments for ResNet.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 groups=1,
                 base_width=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride
        super(ResNeSt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            radix=self.radix,
            reduction_factor=self.reduction_factor,
            avg_down_stride=self.avg_down_stride,
            **kwargs)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             43I��_�����TUc M�Hj13��,odfaf��Y���{w�������^d�$Ҫf��. �-�dw�03Ih�{U�ċ���9g�%U IU}}z޶�|>�Q�k�$3��}l��$�����.k�{z�d�� ��0o�BDk����߿���~����ӷ��b�ea�y}�ܝ6����O�Sw��� ����������G�m�U�=� ��]Uw��N��*�Z����ո9�ϕ�2IFľ�1�Ç�ǺI��r��tR��9�9�}�������z�6H7�X��a���k��^.sN�(5ZA�Y~��>�q��#��β�����y<_�3k-3C���|���k۽?||zz�����`�{fS�	�0����[��o��'_}����
U:�#ՙٝ����8�Zk�I�>}���j��^	���mYUM�\Ns
��#@fJ"]�ض�Ww_�5Z�̣3̫df����ܝ�R��^U ���d���UwK��R��߼5��ry>��U����:�����3���0f��߾~��7�\.O ����/�K� �1�������w�_�����> �L��y+�P���U��;��& �n2�{۶�����c��uMR�K�nw73��l���U�Y���� %E��X��y�x��M�Ț�����𦪶m����8<�~af�	��pJr/$��*#BF-�u��23�Eľ�$�cTUw��$�܆�e`R�Yf���2/��dwg�&��f �����$#��m��Y�ZR�m�Z����Ywg�Y�M��2���}t13�$�{x �3�u���΄̌�2�'�M�Ί �TUݽm��*h�0��r��r�����ԉ������js�*3#�����$�\[�8�w�ߘ����o��0/��&yGw�m������u�	�ɶv�*DD"�r��G�m�a�dU�v�\M�=�<,�	3��+8�ւ�|�30�1����x�ʰm[+suUK�nI��b�QRf�;��F53���sNw�L�c��r��6c�S�m��H�U�ֲ$1��A�{�\$�Pwg��fVU 2k���|���ض��:P�jZ��UZ�|�)i��9�5�0��rٶMRU���<DtVw�9;���=��9od �33��t:].����|���O���������aw?]����o��|T&���>>�ٟ����Z3W>oÖ���n�w�V�$_��J=z�p�h��mû��V�ya���nͬ�H�z���f��YU����Y6�r�9'��,3{>�����y=��S��lltS�#�X���;$XD���@V��|B�^���7�m������\�÷R��@7U�ΖTR=�o�� �m1甴�;n�BF3�l�����ʌAw�2ݽ%�iݸ;ѽ�@�x�f�Y��c�:��5��1���e�i��|�_�Ͷ���/.O�� )Z��f���y5B�*GD�n���N'~~�Q�qk2|
��ҽ!�(�24	3��k��!�W�c<tm��������7u\�{�O`-#��A9�    IDAT��zy�Ew�^���ܽ�u-
���g]���{�n' �M�d��y#hliD��`$��̠ݠ���^�v5-%�af]heDT	3+�����yU�Z�=ƨ*@7����ͬ��BRfJ�d�@�x�Rw;����&Ƀ>|?�K�t��C$�����'˒eǕ`�V�s���6�@� �d��IQ���I��AַrRBa�E Zw���9���ܘ����A ��̜�����ӑ+�%�ݫ
������ug%���;"�]RfJ0���V	tR�����K�f�������m��ݸ]�����|�>�cf�OgRsN���ӟ���7���/�3�X~�q�\k��Z����gfU�֒���4���ZU $u)"2�w��UX7
k�����Hc�9�*"���$?|�p�M��`f���;��l�>޿{+b�y�ݲeً��A�x�����Vw���UPV��� 緯cs�O�Ok!�v9nfVs��9k�D�X�}��/`,����]���Rc3sCwK�DU�1�l�I���b��_��w_���ڍy;���+�r�;+[�udf�Z����뺊�R	.�Y����̼��il�rCD8�Um>�����߿��.���$%���r1�ZKə�1��I�TU�cK23In����$�ZAk�ԟ%��9ƀ����W���t�������r��n7Z�T�M3Hfx���~�����qc���^��ʗ�N��O���Q���&if�&if�YU���:��[
UvDTH#�=6W��v-��݁0�{���Fn��͛�&p�Y��1o��b��^U����0�8:�
�Zef�:�$�`�ef۶��L3[kI��i�))"Vkw;���Fb��� �Cw�C��n=�Mҧ�/�ۭ�P�	 "H��7�|��_�ZF�I�� u��s�t���5Ecd�:;�zܶ� i (��n3�����?�D�3�Z��K"YU�CD4�"�Z!�Q �G��:�F�Z�j�N�S������`f4�TcI�H�~:}���q�9?~|��#�V���mk���&#�}OO�1�R�z?=���ۃ $e.wo��$L���M��+�Ҭ �c�2�T�EQj3sw�%��Yc}3V�$>H�j��l`�5F h@��V��-�F�a#2�j03 RK"�-�n�;�%u�$ ��@ tKո�7T�9+�}�F���f��IUE������{!��Y)�³���;)s����1{�m��ZfQ���ZU$���n���Ӽ��D�Nc���>FD|��������O��)b?��������/�ʮ-&�\���������us���0��I�P��њh��YUfV���$Y�$���̜*�9ݝdD Ȝ0n�	*� j�Z�%���3s��-��E��Q=�޹Z)`D�SU��öF����nU� 3���Xk��ֶm�uMA�����hv7�@�;{nf$����nIf֒����1� ���@�i�i���5�IZI%7zDvYso��s�M� ��$3��4r۶I��u�fNҗ�/�Ӂu���O?���f�>���o�1���w?}�p�bP����(aQ���)ضm��ix�Z�����y�[�a��V�(C�̚��I�(k���O�~�q*���%�O[�c߾̓�/����ZR=�& #�&>#i�VZ8�%��03ȅAH:ֺu_�r���EfZk�CU�A@�,<g)`��nU�����k�p "Ը몈�*��z���=� ��Ԥ�I$��:�N�2S�1FΔ�= I��*�h���3c��u�dw{p�@Ҷm]���`f�i��7I�`RwK2�Zk�w ���lU�1�}7��f��74TUgY	�Z�dCa�� ���P0q��b��8w��5S�.��~޷m3k4os]�$��r�="���*�{��_���������>~x��/~�n����m���Zk�9�Z��o ���n�� ޭ�H���I:X��� ����
�����1ΧSl�mЗۼ\��qࡻ���w�F�GTվ�qy��u@�Χ�1+�r���#��Fw_�W=��Z+��������|>�_�i�/W�HV��Z+��9Wf���>޾3�m��r���ݽ�#bۂ�c����MZ�*3����Y7���o���_��W��ݚ�ٷ:$e��PU֢�;o/��~���4���H�Cf�CU%2�F����.cCfV���������Ǻ�%�$��Z�.�����C2V��62|��n��� D8MRv޵HJU�b����	� w�nw��o��j�T ���ӧ�\Ϸ�,",���A�����{����BC�u��t�^W� �@f���w7I �)"2gD ���rFd4�"�dw��=֚���̧�T+E�&�p7���Y�{��&�]�ĵHFDU�� I0V���Tv�\k��d��g�0��N;II ��pw3�.�}W ��͛�9Z xׂ��$��a!�dw�q���f��jfv>�}ě�o��� ��62��� Pj� H�Z4Pt���Hk��������V%�ݽ�p�ZiBA�����wߑ4�df��1� ��`$���$�4��P����첇9����i[f@Rk`xf��nfsa���t��n�9����mN�BwϙU���t��r���y:�^�zP�UE7���f �Zf��Uՙ�zۆo63Ky>m�V���fwg���1��h
�s��EDf�12��� �U�c�\ ""3ݽ�)EDIs.��,�U��:��� `U�Yf�[wW�c�ef� �D3��I �few�sΈcs��8���o��mۺ�>kQ�m� ����/�z�n��ff�@w;X9�1�.��%Q8���f�Ij�v�6�Y�:"�1̌l�f��43ҳg8=�}���~���}sVj;=����ǏkB��m��Z����?~��9���Ԕ`R7� t�Z��f��,��4����G��|�����6w�s��ü���u^͂dUm�	�i"����������Yi��Zkm�I���\w-�پ�F.�$�B^��I�3�]w��y$�������^ ��uKefAs�y�adkD8 7� �Mٝif0JrZ��(I���#CD�`�L�Q�pr�;�"}�	�z0�GP�L3�����ݽ��h}m�>�%���_o�����G��WOo�~�����O�[�,�Va�t@�aM�p�-	��zߢ>=]�+���"�Ru����\��ni�o�f�E^�
2�ЀG׻o�ύ ����^�+�[��v?�٨r�it[�4ݪ-<�%�; �T۾����l�z'��j]�g�<������v�mێ�d#$uk�wwU��ѫ��\٪63w'�*%1�����$ �Nk���\��@ f��s�!��U "���"T-@�� I=t��������H��Zbu$ef2��fF6`A#Zw�N@v�YU�����N��f66�;�#�%�Y H6�Yx��r�9�����]�R�,cī�{��,n�������������H*��p����?�����1߼y����y\��ݯ/�$O�8�z͇�<����۶E��e�q$%����a�TUsNI�����ns�c��4��:�c�wI ���b����]n���O�q��$3c�N'�"��{D�\���U2FDvݹ{g���; s���܎�dUA�n��f-����ׯ}ƨ^]v'��󳾼\��p#�Z�����m�;��?���f��F�:?u�Pj�-�fl"�9��%�lxl���w���~���gYUs-�$u'���!�wgU�����v� �)� �Ҝ& f��ض���~޻��/�|:=	=o���}#y���^kU��I�ZsNݡF�/�I�nfUef� �3�1$X����Ӊ$��U w����0᫯��� �1����r]I�il9 w�n����?|�p\o���A���l�܎��c��* ����^U hj��Ί����x�nI���,�}�Ef^��5�2��R���%�t:A�J�2�,"����?��8�ϯN{R$U]Uc�~8�c�E7ޅ߮S�t3]+�ͼԒH��,3����&)����m{��y@j�$� �!b���%e�$ �03�����1�Zo޽}��{I�9ƎIwo���O�f&��9�uLѝ0:�v��1�[�2�0�d�&���2�~���d@w3\���� ݈�����H���1Ffv�Y�ARwFѪ�[$[�s� ��H����p�������t�U}�^�1�k�������j�Į*�XU$���V�K�gl���ܶ��L��f��U��c}�;����$�@vKrw�uI �	�s�V�B;L�o�JU餻w7 ��9�Lc��$	@��u�$3`f���HV�m�����ө�R�9i�M��$�n�]U�IR��u��|~��-HI��\��/��m[w`+h�yt�HJ5gv��sU�����Vuf�N���O�il���w�$��e�$��N����o�&s�{������ˍ>,|^^:�������^�Z����� 2s'�GI���R�J�FW�]AR��U+�3ӷ��!i�K�*"�������9�m��GC�r�6�N�t�.yk�\4�檙Ǿmf�j���T�{�6�4���� (DDU���T������݌d��w���!	��[M "���FZ�QA�m;�J �Na���F!fݎ$��}l��m�u�1$�������ͬ�Y�E$��� ���\�*��Q�U��_��vc�����3o��'�v�?=��oϟn����l���z���׶#�0�̜�Z>��O�_�I����6�
�@��N��,�vb��Я�tZ�~[����l�m!?g�]t��n��"n�Y%zY7H3�C�4�YxwP.o��t������3�Ma�� IM(H��:�VUf����M�9��I3��D���l����K��R h �
���Zi�H�R���nI R�DU�δ��@��u�$w'YUQUv� �n�F� Y�� HJ-)"�[�����q�3�ɭ��9Z��p:۶�z����p�e�p�sR��κa 
��8��^k����ի7O��n�O/��ʵ^.4�_�^�����[�o~����������'V�����O�,3��k_�zs~:�\k�1猇}����2s�[7���df�2�r�u�$R�߼:�N�=�1��Rf�9��t:����1N����ط?��Ç�KɈ���@�ݿ��+I�����KW#��vwUw�t:m�$��z���A��$�i-���=ݶ�o�a�@���c=??�,��G�<���_�L�珟�/W3SC� l{�5�g���5Ơ��朙9�>b7����_�����u�����J(I�ff�jI������w?}����m�2'��4��$ �2����ݷ�N2"���N�W�޾E#3o�Y¾�mۮ/��Z���X��
��^k��̪�� T����n�y�Y��\-77y��I��@�Pڶ�~���t=n�^��ۍ��A�*t�|D��������d�owQ�y[�g///U�H�;��63�d�c�w7p�IB�*3��������c͗��̬��i�c`f�ZfA���u}x�>??��F��鼇[ (@w���Vf�6���v�q�4w���gC�j�3;�x������}�O��,ܗ���I3�$��23I�s�AgU9���>�}������*����n����h��έ�H⡡��ü^R�h "���;���{w��� HV��2ӷ�*ݽ��ݽ�r����[�Ѻ�*U��]IS�UYU̎}c+��:���-��#��x'��cU�$wI��q'J"	`V6t���l���$ED�� u���WI ����mf[�ʖDҜ3�d��F���!�Ђ��H���vf��k-���r� P��YwKrZCHrwIUE��Hf�����@w��L�f@RCUef$2��w�$ � y��Y�f�n&�i�mDDw��Z�d>l���lU�ݙ�N
�s5jU�<�LI��\��N�Sw_��}��Nt���5�]R�����W�����xz�~��K¿���&qf^.����/��pG�r�\�	6��@���fX� �4�2��� ש��	�2�1���93��"����]c5ƈ��>�E�t:h��s�-�t+eܬ�$9U��� T����o֍m�V����|UV#ܣGU���0o
�o��K�s� ��d��@V��H��F����$�
��"�}�	2ߌ�p�58b��II�-7�YgUulaf�%cw��.�T�P��[xw�p2 ��+ۈ���~��9��ӼOC��+�J5���/߾}+�v�����O?�R������r����O���oV��ח��|�|�xF2t6�.b�k�w�6���Lgq�ޥ�M����i��浱s�z��F��
]$�]*�p��n�1�*�� �ʺ1��m=�n1����+�fr!�ɣ3�E�y��.��̘Y��4!��ef2f&���ɪ
#V���`��*3�[w��, $%1;Uc3#	cw�;�M��d���[�amf�U/��N!���ݻ@$h���Ujw烪�݌�63|Fw�����̔���4a�a �x��0��ݒb��yl!#i�՝�}؆N���ݒ>|���r�*
��̧�������Z��z~~^k��z��y�����4������n���~��5�:.�Ϊcގ������df ��������$��j
e�ne�� ����6�G�~��l����6�쪚��O���o�~��-�/��O���� `f��]zp2�����|z��/�U��-5��M�����m�IZk�n�#Ww�{U�DuW�1��<�67�Z+�����`f9�ry�^�	��O�x��[3��?|���Ӌ$ Rm���Q3n�I���Z�><�c����o~���gUgvUu���Z�=�D�����?}x����O1V�|U�]E
@U�\G�}߇9Z�"b�^�~M3H�nUD�;�k�̼�n��yw��J���t����朒"B��w7I��(H�����K�7�XU۶��ޥ�7���ojM��۵aU��O����6u�:�$ͬ��m��n�"�īW��f������������Z$ݽ�̌$����� "�e�mvA����ސ��a�ݷ���Sf�9U�\�_��mt0�a�`@��_���a������/^���tw $�\$��(�Df��,�$�YDd���$��b[k���ݍ���qOOO�2a߶��'����W[f&�Ih�3�lw�n $��dlc��I6t~����+w��0���uG�I*"��p�JMR��S�����@D�Jm��p�Yݽm[w��U�� ���l�!3�if-uH3�QRf�^�c��if��d�U�\�H��X�::��� �Ya�>��QX5��@3�$�*�;Ăƈ�8 �����ܽ�I�
��S0�캳�gt�n $q�-�i"H����Jw��$0[�D2�
 HV
�$��f0�Cvu�����i�����Z��ݒ���e.3�s�YD �n=�3 ��薹��ZfF�dw��pldw 9�naf�m�U����LrΉj�����Q��9oE�\w��жmk�1gW�YDd��}�6 e��3L$Z�-\>�����믿��ׯ_�{z-3��q������~�G��j��h� �
s�H�s�����]U
�g���84{Ωw����>�JRf�IIh�}[��i���l�4p�6��k��ڶm��Lz�����Xk��f����̎� N���ܶ�ZB��@W� ��[Uuw��3�w���	�Ny��[ؙ�,k�pG��I")�-�L�̪
@0
Xݷ�Z����]��43a�ٱ��"�j� 	��A�̪J�2���.5�zk��,݌?���\?\�$p�"�������ݻW�����Xa_��m_�?��wy���nO;���~�����������l`�1���N�Rc�S��q>����~�W�g)�������K��n��ׯ��KQz��e�}�T�z�^ �D6df����˼9Dolf��FO�l�y�t��<" �.I.il[��Y>BRX�. ���btwD��$�H�T5dfnQ� �pg 	���ݱo̾ժ�mc���&*|&{�L��:%� �$e��-'��    IDATE҄�&<�I�n3 	���U�.���2��Zkm1H�Cf�dDH��f��y�Y�� �i�3s��Ь5̝Vkp������ǈI���Ac���,���1.��Ǐ��(e����{��O�c�z~~^kU�����W^nW�c��y{���_~q���x,g��1��*�z������s���sN=l�f� w�Tw���L���)�s����O��ֺ�n��ݷ�-֪��m�7_�����ݻ���~&i$�!ɀ0w�7�� ���w���m���|�]���ӑ�g_�G� �����w�ke��OO�7g�PU���Gɺ[���R%��~���y?Y������|�;H�OO'ל�.���c�� �1VWf��F��7���7��9ת�<�J��.6����$��;������Kк[ƪ����="�[���1�$#s�w�EU�JIf6<$uwfW�c���q\.�f��;>�zuIr�����޶-��LRg��z����7���$f����]����/�_\�#".�+����w�=�0�+����Eݡo��$Һ{?���o�G�1����9'�Ӿ㡪����I�a�뎱�J5 �� �K�1H<��2�m?�C��s�m�~��/���j����*w_���ۇ� ���_�u޻R�q[�!�DPV�.�kD�\�"�  �j�$ 3`T��N[U�*Hf6�Z�Ý�m�N���$�u���%�n�#3� i�gN �.)s�Uef�������c3��23� T�]a�;���0�ZRwf��"B)d��-�9oq�� )"j%I���o���dDT�Û�D���VU�ܝdu��A[��>#if�k��̠"i`v�uGr�g��>���P��;�$T�# df7̌��&	�t �=+�]\kExUW�9�LII|@�� YU��| @A�9���; ���;��X�Ȉ��D�@w�@wܱݽR�mf�`f���%�U�K
ݡE ����4�[�dY��1�\ko?'NDveVuUߠ�� `����!!~��Т+�:�2���{�9��û������*3#B�#��Zk�� PU 2ӆ�@�W	��m $�;�Iɯ �NS��&3�#.s�(�/_� 8�-�@�`t�����֪�εV����0��j 3��\��7�|������ DDDw�)�~y��o~���J������`=�