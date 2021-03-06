# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import MaskedConv2d

from ..builder import HEADS
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead


@HEADS.register_module()
class GARetinaHead(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(
                        type='Normal',
                        name='conv_loc',
                        std=0.01,
                        bias_prob=0.01),
                    dict(
                        type='Normal',
                        name='retina_cls',
                        std=0.01,
                        bias_prob=0.01)
                ])
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(GARetinaHead, self).__init__(
            num_classes, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        self.feature_adaption_cls = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deform_groups=self.deform_groups)
        self.feature_adaption_reg = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deform_groups=self.deform_groups)
        self.retina_cls = MaskedConv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        cls_feat = self.feature_adaption_cls(cls_feat, shape_pred)
        reg_feat = self.feature_adaption_reg(reg_feat, shape_pred)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.retina_cls(cls_feat, mask)
        bbox_pred = self.retina_reg(reg_feat, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred
                                                                                                                                                                     �!�j4�.2�����3%U�Օ��HW��/��Tvu׎W����^�TX��Q{�P�@����ي��Tba�<��ibl�E�?ĵ4KQ9e�y�3��R��J �M�,?/	Gڷ�L\�xgg_CCO}�p[�lG��D�A��O_,=(��� ��XiM�2ݥ�N]I����#ٝg2����K�j�����������9y>���s�|�<�UC?�'�AN��Mڥ
x��,W���@P}�xG]�hw��ڢ�|?y �0�0=g�cS}�C�g�Nk�
>�c~P?�˿ �����h��������`�po�����B6�>^��>�@m����� `h�"��<"�Yh8� ^�n�﬛�, oU��ם _o_ﺻ0�ti-;,Zo����1�B[�EL#�ǇeC�r
֖Ar�;��ف�]����˗wV��;�G�G[+�ګ�ZN�$�9�B��3��e ��"S �>�~�Q��y �Ը����4 �ٟ����.D�@�LΉ
��"���c#B�#��#���b�� �H�+�i��7�@���@�C"��C,,,�o߾m�6d
�n>�=��&T
�FeX�	��`�J��\p������y�9$3Z\PRB�f"���C#"=�|1�>, 4*$R-u28h2O�̀�����}��W7߿�E��~���.�p��r��H�/ڑ�ϟ�Ș�/_w�4�q���z��l��[��z39�z!�Wf��'[��_���I����,M�H8�ݛ�-T���Ӵ���٭��5)�2>[7�u�6�
���{�����Wo������/�ǯn����o�Z���ć7C��{�� ������w/�v�?{���i��[^ݮ����탺Ww�>��������w3�F;s��s�};�����/����S_�i�-���v阗�#=�_��yj������,q��;DxH��4!.�����ӥn���+g
�T�ʊO�M ���d< �J&�!8,��'b�,�h,2���-��}�%��4�oK4305�;��}
�o�6*ͥQ���~*��B7en���z�6j&K�`�t��5���u VXR�jkM#֒iiaa��ڱ-���an6�1��<v7����ً
�K��UI�˱�D^��,a�jE�~�4���('��^��y���1eA��<˽:��L��;K6�fc��jy�"�0�Ɩ�P��Q�i���pNV��Kj���3��Phocch��\&HX`c����� u�����b>�F�R
��I�ebD���O��i0��-#�����B������ޚ��47�?�w7��C ��U0�ݹ����{�y�32(��A�[S|V�
�/�_��3D�����ڃ26��#��QI3�0�C�T$\;2�0��BH@/m�h������b0��m35�%ZX�?��s�������( ��E�a,��L�L����0|/�G���@�9���-hD���
�`</�fF��@�Y�X�D0�` �I,�_`�t]B`�/�"}�r%tjE8l��O� �5,p�
� �i�9�UZ�d���P�l۵u��/�@(4S�>3�.8��Z��,�Hשh��э-0{1��`*g. l��B �ܰKp��ԥ۳Y�YW_J�P�829��|��#(�
Uds��A�<�I��f�#�8Vl[���:���7kP����U�W��kh�q��N��w�թ�t���ȗ�5������V��p�A��$ �I�7>���Xt,�#�%;��r1����p7g����_��?���?\.� �۽�О]q!���ǢB��d��>#�CZ\pR�������+s`8�r[�Z �� � ~�D(��	�Ӣ���]�x��ͩ�: �ϙlc�����������AW�ʎj��t��Ç'k�s��YJ7�L��rI�f���*��B�F�=�d,�/�~���B���>�d��R�v|;:�5|8�C� �vq��+�"T��+F��-օ�ըذ�;�ưG�h�R�e
����H�����U;�!�*����`��8��	qqf��"å�ъ�(��`�FEru���`y�1���=<�~~|���Z� ��X����q� ���]'�3�KKGB���~ғ	A�'S�SbUlZ����; �	���t;��V��\��Z�vL@��-Mo�L��޹�T����ޭk�S ���N� ������Ǭ-1|k닅���}���N�E���%?[[�=7�61��65~oy�y�?�o��&�]�}yk���RQ�n�_ �����K�[��8poy����.Pg��rm�ec�}c���R'�f��RN�ωĀ����Ѷ�����:�B�;�{�[ꋳ�b}���Tx_��-�vU��4]��^U�U[�S_� 6�!0��oo-/<�R��^�Ņ���Qqecf�Vh(   IDATnqd�;��6�յ822�5R�_�� vS�W����ū�۫.ל?���:�� ��ˏ���U��r��[�_w;�wJW'������]'�}fF��5-)'>.�^(���XR�Duy��EG���v6$�;v�Œ����6HG[Ko���howOVF�ܼ3�KN�,<WRZz���¥ʊ꺊ZHk}[GSgWk����k���ff�g�������ڵ됅��k��n޸����ѝ/>��������=_F������{�H���س{SO.>{������o|����>|�}������ϟߚ�� }�ߘ�k�i�v�sm�kq�mv�iv�a��j��2�ZVy.����/��j�(9�t�����'�Ó"=���f�L�*��Z��o��_�j��4�0ܼ8�8�[�YS�1�ym������tף��[�ckW@kE�������"���tV��+�_}������ �e'k"��$�3�F��>�w���52��� `S�e}ڹ�S��26�>2q�h�w߽~����G3���Te��z�.j�2.��sEm��P�윾Ǐ_PЍB�P(2�<����2�$�  �0  ��o���k۞��_����ry���8��a!(v�ȶv�$��_���¶pN�Baq ��N��;��; ��գ��;�J����H�L��S�;RYV�Gcg�_�s��|g�Ts�xCmCF�?�kg|(/���сlx���2;<8��UC#Y��y:6v��<��<��>�Z�-� ������C��w����H�r�΂���ʚ�d��A����|NENb��c�ʲ��j�s�j�J���;��/�" ��!a �I<8�� �h4MU���/\�t�|q��x�$ ����[�+����銒+�ּè��;���E7]��>{�TzfNb���M��h ����h�~ފ _U��: �)�S �h��M���mh)!Au��z[@��-���� `��H�D�іձ֥���ϖ�O�&a���Df��(��e*H8���\�""Z���r��b����]iQQ_�~}wu|n�ua�e��j��j��1?5ik0 8-6=9*5.4�S������MO��p^�ɬ���YI�iɐ䄴�x�uNr���UZ�1��7:$����� 8"2.((�L&�M��	8<�k��������â�bј];v����}��믟|�,��q����ё��a �p�p�DI'Z��E�z:��yHȑ$ǤoߎJOO�ÿ<���7>���2����?�O���)�"{�z�� $��qӓ[�_����� �oǞ�ľ�֤,�V��y�B��u��/�L��ZY=�C�,]�������V���k����_������_�����?�z�_���oצ��}�BqiX��>�}��\߉�s�=���j�?����Ӄ?����߮��S_�ľ�[�����/����>���J{~��ݣ���Z@�3��K_ܮ}~�ma��@K��,}����7]�߯^��X�n/�]�����'�������{�����^�{ ��T(�!)D�@O��mJ���]��l<ޖH�R� ��ݿVd"b���[S��b�	�41��q�Vh�ws�O��}�8�/З�#���B�ܷ}ס=�ѦFR{�������K.r۩��r6S�f+�L%�-cХ������!�
�c1[ ��Hmg؀���6F,C{��tDSYB��!�JT٥h��aN属	��x ؑx�gz����6
u�g��6��n�o*q;#L��:1�9�Pb=�N �a��7�oqx����4=6��ٞ�����2{�d�s�����CDs39��K)�ptp�H\�R 0D��+�-��( ��r��_ o)Y��S�n��B�1���3��Ƈ�L��42�[�!��B���ۭnޟM��t�ޭi�[ ���%����F"㜷 ,�r>e�? X7>�b
���l,�!��`��a:S7hن;�f�< 
)pE"�S�����H:�10�_$`<]�-�kGD*��yPo��uvQ��\*gl���݊�����1~g($�����hC�}?�sZ� #�K K"a�T|��V���h42٘D�u��[��J�G��1���h�u�ZD N2��t@(�rT1!\���^.�Y���o����4�=��8D[<b���H9]$��6W��&�ITx��~,���Ucd�4�i��2Ec# F�'/�2]u.�̖o!�UZ���.W��.|�.�pf��D��bA�[�u�C~#W������m���x�%Hț���,��C���RN�n�]�Y�9	����e�����d�9N��]��T����eX4+�Ж�vRq�<	�`R>����?����p��[�|���`ft�F��($OwW�-��$��l#���A*���ki��
}����AC���N���`��+G�� sԕ��8ѕj �ř����>�n�Bw{����8(�#�Dæ��~����{�v��H�ᇆ�j4B��H]k�܎
0۰��R%K�@�SM����d�1�\&ӕ+�S�>�O#�y�ܼ��ؔj��gOw������((@'P��l�B��2�Z��5�DL�Ih�Z[����	ѕtv�x	�=�``1�J&�:J��J���ǋ(�"=�@_Ow&��_��+�Ņ� X"�����=\9�j���馢{h�^NlH��]��0��.�]��q:3,?9B�&U^8	 ���(�I��r�p<�0%�z*%AmǱ�8�xm�W_�zv�Z\�GI^��k�3�����'z ��L<�qm��I ����TP hmm�+.n*)�2�~u|urdq|dyr��
�+6�z�p��n|��{���ћ���ũQ�ӽ�o,��{�����:��]�|}����;K=��+N[�h�9�{s�_�*9��9�7?ڶ8�1��m ���Nu�Lv6w��YS�qd����e�r��u nř"���vgM�Pk�@sC���@���At&3}��n}T<�07��}}kSSp��VQ�����j�RWv�$>(�dnĥn.���77f��N�\�2��Zu���փbf���t�"79UfgomEd�ll�T�J�`�$���u�8�����w�LL����Vל;{�ʥ������ܺ���F���}ubl|fb������ ,���;Ϟ<z����w6�߸�����4;������������޽~��壧n޹:8qw����u�������������չ��K�����Wg�o,_�X^^_\������D��@kC_s]��ұ�Ks�	��p��ق���gN�d��$�&E�{������^ZG?W���UulH mv`�?�ɼ�:�g��ޝ���@��[����~8��{d�v�i{6spʑ���ɿ>^��z�������剚�s��#-Ȭ����ɮ���ʒ��ɮw��Fg�;��
2�ʊΎ�,O�å�[��|������3gJ�/;U���s�t��'|u�9��&�#J���8AUQ����WÃ3��ݓ�_|��_�����m���]B��C�;Pf;Q�(�aHh���+<Y���ۆ"@H6%������Mr�H)���]�t1)o���L~�pa�DQ�dљ��Kf%�-8VQ�!��!v%�S�ԝ{f�̙֊�!o���w��[�YW��U;Ps��dDǘ��a=�����ﾺ��:�SM<���g�r�lq`�ͭ�W�r�}�C��G��cb
�3=�J��a��q��=',�x۶�s����׵2�sc�/���B���t� 2Z���k�J U%&/�����+�It&۔&'����?Y�q��XnqHLqxdy걬������������
�<p�qv��&���2�&�\ܻ��ԟ�X���u&;;/6�x�~BP����d� ���lyA1O9��y����)-;s<� =�0#-;.�x*} �*�\�n�A���Q�;��iC����Ԉ�Đ8%����Uy*����Pͥ����5%�u�"XHkΏԗ-5/�g����������xK}s�O!w���tk>��2�   IDAT1�Eg��d�m�dGl+���D����xe>�s}-�-U�-s=�Y��r�`ZZ|FV|VFlFJt �M����t"31����r� �9iy��9``������!q�Q!��!	�ѡ7"(2�?,<0"&,������`�%���M�I����$�'2�1mX�O�����K��[��+�0Ą���~���wݽ)�x_�G�o,$>8�vК2�prO�K�=z����O��؆BEED��w�����O ޤ/R��S �ى�F �D�{��ě���f��~��œ����߼��7��;6ŭ_>����_��T\Y�7����U��+E���.!8��]��!Ľ#k�������w��>�����3���}k��Y���ԑΏ���ʹ�x�������� ��溄��@��Ǐ������ԗ�G��p�7��������￘�������/o��8T,3�Ø�{1:0�'h���� WG�ڽ��O�?�ݧ����~w��o����o>[�棹�_O|�N��