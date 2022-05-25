# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
import torch.nn.functional as F
from ..builder import HEADS
from .anchor_head import AnchorHead



@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

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
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ��ˑ�%�O^={9��wƟ��?���������������}�6A$�ur�9�#�,B�,P �B('�"�1��1�Ʊ��vO�5���t���n���vmq�q�;��{ך7�o��O�Nծ]U���?=~�'n���[����ƓX}�C�n=B[�|��#`��o��ȵ[��c�\���K�^�0  0Z��A�W�x��/��~��k���k�?MK�t��Wo?�x�7޾��/���[���׹�%��B��(�R�U(5h�
-#5�a�~��g��XL�G�|䗿����������矼�,������G~�n��Os?�j�G���||���;��ʏ�� \����>�6���T~�����Un�"w���w����o���Wx, �xA? ��on�^��kj�5;��sl�Dש��gώ�A���z[ �CÝ�}��5Z�2{|�s�6m�}����c��=�C'��[{Z�;��;Z�� �涣��c���= �#-;w���[��R��``6�����Ҍf](�]�nՎ�;�5�kjk>�҈��m�n>�u�����6ڶwwm�����v�D&dsS�S�R��e]�l���;Z[[��ۻ�����׬�Y[�~c��;v�ߍ1�~��M;�Ԭ_�x��@$ĕ�Sxi)'�鷹vo�3�� C�; 8��Y�,R:�_R�+��+��x��P�3#��x���R_���϶��Fw��\!�'�d���cv� �X�#){4q�C�=��w_u�u����2��Dq¹'ήy�{0��?~ fm��ѷ�;�_�P�;P�'�����`+�c�'l�9��[��) "S� �+0�#&C@ �[ `�u��� � S0I�c�dE�#��\a�� ��e�R�`�����23 �,ֺs�4/4 2�F�#���Uv��,<��jG�D\�j�C"a�V�:�k�.�DJ�,@EB��J<�L �h�\S@.�D*�
+ ����"}�Ԕb��|y&� LG刈���@�*�\F�_���*t�hS����g�j��*o�4�H.�P vG�`���oDa�B�_@��]r\6���P�Uf��0^��Nju7B�n�*�wu/3��N�Z��ؚD� C$4S%�4�����h�7��RH3�*S� �l�:�F�шe
�QԤ�	��un�5lYm�zi�_}oL��+K �j,|00M�=VK�ɫLCmq��苏h�HL��4vw,�܍�y�y��* ��C@D�'�;V��yM@KC�I�"��CwM+�fтI�?it+%�h�e�A`�˕z�LçC)ӰPg��m�p���ɼ8 0Eh��Q槡�4�S�IM�}�鞾�vf�Ϡ ��,r�S ���+���m���i����gqf�sc�e�
�D��~:6@1E�ωO�-I�9l��R3
��J����iD���UX�zY�$F�K���(�Ć�/>}r�Ց	��㋤4�0�yݐD�f���hb6����kJ 08n�jA� `�}Vz,G�N1��/:X����Q��2�+�����< ��@�]�K�y��#��LlV� e�3�HZl�-6�ȟ�E_eRH4���)Go�
�|�0HX�⋴(�[ �~�d�#�W'���������[:��Y�Wg����ܺ��r��G:�O�D�<#7A-�!I��rgH8��B(Ѩ�{̐2�˵ce�T�&Q����a�Y nطv�. ��}u `h����J�!��%�8�P��В���k�3K��,��)�gk��g���S�XMYc�a:K��";cnUR�%��20��""�������wS=+)�R;-`U���Ժ��Mg*�Hdl�ݜ��\���.�]F-� jZ�������r���싖V"��R����4i2��I�cA2�R�� `�W$֑Z�f*c�ޔk �r-4�ԍ%4�"7D�����R;���y@���%ۡ	�^m�>-A�4[1�oe���6�-��صܭ����f1�A��C^
�Q�_`�׬�9�/\X=oͪ��U��+-��K�!�w�,��N��߃�J*
�6�y���b�P�`	C�Dt5Я:�J�Fqd���+�V����U,^ .�*,^P�ͷ�R�/��u
�����m�����Vn�#��6֮ ϭ���WXVZQ^ZVA�|+��p���s�,���e%Eu�j4��=�s���^:�c���k�TTW}]AnIѼ��6nشe��[�ׯ�j��֭���T�{o\W�~CM��Gz:6���b�J&$�%%����)��t^
��d�!EY�P<�K�ҙ3grc����F�v��go>����ly�lh_C��s>�6��*��-�O��~�'���~��}"����?�F��ot����I�>q�&�%�5q⸝;��z�嗯�8|z�o�`�������ǇI�x��9w�������F�J�G/�<}��t��(�0�����z�;����]`i�q{���xKOڶ�����#��='���>�=4�������Y=����[:{�N� >wv���KW9y�����7l�108x�ĉ�{���t
 |������+�=�%S��?�_$��5 �`J�/�|��O���so�����<����O�cT�ը^%�:��n]���#�˯_A{���^~�2t��K/ߺ���K�^���v `-Dx�&>^�!3K��x0����K����Mzݭ��0��D���{���-��j�bB`㫯=��/o��_�W��qe��{/pW����B��   IDAT^�g�!<�T*�jhV|����y\�<z��|����~��/�y��
UUOLcOpN������9����������𻪏����[	����������;�#�XY�_~�7/���O^���|����� �������*r7nZ�m>������߷�о-�n޳c����w���>}r���s���`_�I�8�<���Yݰ7]k��#]�C��'��;�v�iop���Zm>���������u���pӶ+k��[X���4Fm/<����pX33C�V�=t���၎�Ν�wgegk��4wj��2��`�[�6�̖�T.[�j��[653��M�wl_���~��[7nھ�nc}���.[�S��5��s8}V̤i�f%��XT� ���
Y<fo�ʷ�y�JL�]�<k�R_�R'T��]��[R�ɯ�畹rK��lK��宬bR�Ѹ3���ͯ!�=��2�=Y:�eF��ĺ����|�ek�Q�}��$�Hk�<��'�>����o���?���|�}ܘ�eG>~��,����?}��z��<^Ν0���w���O���'L`M�4����g���2�B:�_���������+�ޱ2HL2�h��?[��r���0qtu�f�~U�`�� 1��Fau�:�d���@��Yr��..x�
"��E&���	uD���C�%̶?ǌ탷�k�6ufh��ǘLf@f	K��'1&k]|R{�'1yDX���5	Lrb�OL�-!d�PT.�
Ia$oq3˵�2���W���k"���x���\o����>���16��FDklH �a�jh�mKL����� m���r��1V_�"�;��5x�`����j#	��K�*���Q�MA�GCǊ�*ƊEy�0�̙��@bz,:)�8��������r� DG���"��TQ�HIa���QIt����'��w����ҰR�f�Z�I�-I`j���&�?`f&1�4 ��(�"�ޔQ�L�%n�M!��K񘖃��0Z��Wl��@��w�R�t6���}��*L;1l2tG�a� `��/׋Q�d&r,5�_qR(uG-��FS�EŴO��>Yz� 1>-�������MΑS�qbX�g���pߴ䩓fM�?n2�&K�8eZ�,<�`b�(%I &��f) �'�q�u�t�C��I W���q�3q����ݧv�y��c�#SCK^����l�``L&�����Kf�twb#Gj&�|%u �x�����	��Y2�L����*�vDz��$h�8��ԩ R%�(\i&��,����:B=��g���{�r�}y��2�3�h�Ћ��81)�Y4�4`Ы��l>�\+"��]:���llh�\U:W��Ӥar��R��:w�E&��%U��%+w��;��]�!�O�s�Qx~n}��чO\����ybW�\7O9CnH�;8��4z��)6�&��Vc��\�)��iR��!5fj�e.{�b�2W�̩�]Ӱ�f���w�"e�v�(���ФF�%��/����3�:�V����i� @��|�9�24EHd�1���-\���x��;���.��R�eRB�)�J�ߢv�>Ҵ�,�0��_"|��JI7����"��<�Q䓸U���U"r*�kbtS��K��435�b����N�Z��	j@��H�R��$,�
8F��.2̑�z�4�5�W�sU�"�5�i+p�I����Ћ�Xh/rz+��4�_*u��Չɚٳ�1��8�%�&�&�����e.E_�M��D��o��J�V*0j��9�leA�_�?bPe@�pȑ�e�[�C��ϯ�)ZYYX���٤��c��l�,���N����ڄiI*
��V�b� �c9��*��c���������_��j��y+WT.]X4�lq9 Xa��fs��ګ��_�M.4K o]���� C�V�\X�� � \RT��N������/Z0���#ÃgO��<;���ކ���W�]poaEْ�+�6oܴs��m[6o�_�������a]}=ц���������h���u+�υ5+ӤB ����%&�p
���N�
�"9��G&���D.��%ɬY���(� ��|�%
�;��8����[����:�e����oU���8���r��J�ۯ�����Gʷ^K�X���Ʊv��z���׮_�>��~O�m�:Nc�I˰h{���F�\�8������y����������� :}��������oq[���Z;O��p�ѳW S�ş�����'9�O�]�8z���ٳ=���+vG ||h�̅���t���SgN���u�m�������=px7 ����_�N�`�] �[�����^z�����B�ޏu<s?fl�T?�Jʹ_�u�����K�/\�q ��+��v�
E_
�XQ$�75��ި����wI�!���Nt�Fȷ�� `�ןy���|���)*fr`�� ,���^"�J�HK�7�TT1���7�n5���7_��k�޻���~�ICv����q 8��/�>�
�1��� ���[
�� �{���O��6��2�٬+�N��Oׯ��`&��#4�n���+o<���G���̻�p��۲k���Tf.ZX�|��5�,_\�dA��������#������ힷ`Q(+;+�(���	e&s��i�8<kt����͛�:x����D��P�@kO���cM�Z�Z�H*鶎c]PGO_go?Z�������^�����\$�p8i�������艓�`���z�Œ���Ԫ⊒����غs��m[wTͫ���es7nݾ�P��v�ݿ}�n 0���w�݉-�_\�[�'S�cg�| fʴ��ӒZ�֢5:��_O��ΰB `W��
�2
�R3�7�\Xe+Yh-[l/[����m�ߜGn��a`;P ��"����g�;�. �?���48�� ��6 ��z�� �-�GX'M���rZ�{����j����<4U8�屛��4��f�o>�x��쮁��%HfM�x�����|~d���'�� ��>����@�T�2�.ai`����Bt�}�.�H�C*3�K�i Q��w�)�B�L�7S�� 	+�j&��uF�%a{&E�3]cO�����4�� ��@V�ɛ%��� gH�ucG�vJs,�CSm�0I�%<���J�JaMW�Ru���!��n�%$�g�Y
{��^��WD��B3cQ�/��
$vDD��2���,1�ו#CK@��_W�D�JU�RL^�`���w���g¡������A��f�-4֔Z}�F����4N���Z_����=�d*�Hڠ��T�\G�d�iA_Q��k�2[�<yׯ8-�O��)�BL���48�0 ���f�Fk�&U&	�E1�I闑�TdՓ�4ɂ� �P�J��%���n�'�4�r�B/�Y��#���T�! �3���(㎋�FK��Z� =���)��)9�H�������a���c��PM*Eሬ>5SՉdV�2���xK^=�զ� �i�d^��]�#J��4�Tϴ`52�\Y� L3`�s��2L��وcϘ�� k
��:M�R[��^ \7�uB��nL�/�) �QMω��6�(�+���@C��&6~=LA9�/� �y�8
�����z��'�'+��!w�$�é��� ø�Jt� `� "^�&�XԮ]|�c���"�i�i��$Ib�x��"�	p8�P�K' w�y�`��ԙ3z�&]�$2p�3,r�~E�&�"Xh"�'Q���M�?=aNA�(QcQ@z�Ze���z��,�����i�$00��e+�) ��c�*����6*s��� �s�Y]����R��Wf���5��}���ֿ��gk���]�{{G��:��p�閮�G��k9��p��}�[��n�ݲeۑ��jW�\Z��8�"�[)]ZT��0�"ca���-˶���}�a��9����b�����F�=U��J�g }�fȚ4C`B�/����+Z]`+���haB�.�s@_��,��]�(�R�u��}�:��8��2�s   IDAT)���n� kM96g��z�Sx��H��Rh���:IL�C��b��P`�"���[�
I�k�L�QR�vQ�M�>p�
���B��D$tD�����S����S� �ϙ|]ji�&�m�&��9�T ��.������5��*r#Z&��k�c�*�
�%�I"�Q�b�^�7(��b�A�Ώ��0� G�UV�^Q��

�g�|0��,�Z���h��h���_4��Dݡ�7���D]�����MR�s�W�Y�z�
��*�����,[D�`Q �,YVG� ��.�ʣ C�-�k����l\W��v���V���U�������W�\�mim>}���k�֭KW��Ȏȕʼ���9s,[�x�u6nغ�v�z�~���)o�P����Ұv��u���B��j�+�6�źo��dp��$N��J%�X���ierIW+�͈�ə1}^N��h���|�O>y�'-W[3ܞ�b���v���u��+���W��;/�w_��@�y�3H��?)>���ɧ��2��CjTN��u����{���j�n�=�7|����ѻe�[{�Z{O�����C�.���3|�K�;�;q��8�O�϶������Aj����;u�alj���C�=8:|����$�4;4�����񡑳 ��;;0| ��?������������|���7^��O�����^ ���]z����o���_�փ$l�����X�2 ��#��9�{)��|ig,��*����#��K[*7|��e*ڧ��i��?���?��Qo�T� ��T��|!��H����Z�RM>* ���Y ���ϼ��Mj~᝛����1����f9u���N��/�����O�T}�ݼO��������/���LTpŇ�W~�}����pQ�&�?i��/]~�^���3Ͽv�[Qn���\�o]������s6�v<��!�T�s+�X�d�5kV�X���0gѢ��ƽ�O�8uj��qբ��HVnq�]P���_��>+	�<�l� T�XU۰���#m��'���:����-�]��=ԽmsG�����������c�k���PccIYYFf沥+�Z;( o޺)'/;7?H��H��������փ��ܽ�j��H�ʵ�v��u������ؿk��M��V�hEu87�s�Ϛ��7] ��F��J�86b��h!w���W�wX�U�˭4VY��[�X���%�Pn�%o��p�'RJ*�q�@���@JGH�S�.��ϫpgX=�ϕI"E�g�l>ӸX�Űy���~y���	��Ǳ`�w�T�ZQ�k3��ao�NH��2}<��b�Mfsc��D��5k���4����:��0�-~�#�����24	��/)T�=R�]�ٻ����8p�X���y@/Зd�է�)�f��y���7C��x#zO�>���y�ٕ��r��@h 3$���xe�dp)��A� WG��dQ��D���4Z��0�Q��RVM��a`�&��zM��j�hGi�B2s:��?�Vv�x_��qs@�'r���Bロ F`
���m^�5Ri���9H2E�mai�W���;S1|K��B� x��?3%��t46ex��);���I@S�]	�TF��BGwQ3*
�w�9��*�Uf�Q)4-���-�j�R�\��D]�)LRk*�H�X�*C�����19�f�J��Ӝ����W�!!�4��L�=1�G�8��
#�A_�Z�W$��©���t ��T��P�g�[�Gm�Q\�w�� ���I�4D_F #)��O?bƘ>I�Eɓf�����t��������x�Z+~���8��g��黛�)�q��r!&��&��Y��|v��Y)1��*�#'���E=�ij���O���X8�N@�]ќ���Ӵ@t2gVL�}�b�qi|��W�Ϛ�����qp���/�+�~[za?� ��&KM�$'��9�f�͠��&��.�8d4��S��!��)6��2�����NN�m.6$����Ɣ4qn
���)Ư"Ixf�^��e��ӻPK��|[���\Z�W�Vj\r�]v�ŝ.�r���Ljze+RI�<�H�WYe`{{�������`����)
;ߕc����t��P�I��T�"&BmZ�$.Y2+Eǖ'����4WI>�Ԧ�:�[�49��@��8��3��$O�rZr�-KI'��Vblh�J|ً���_�d�²��i�4A�ح��~y�`����c������>|�dSϩ#����t���7r��dc��#Љ�-Cg[�?��ܹmņyזR�-\\R���pAQ���'�o86r�{��y��X[�Ҟo��3�rg�̼�1W�ʐ�j�6��ZDE��0\��$P�!���)G�E�20 O� ��@� Oj��YEe]P.p<Ƴ�&aH�	3��L�:d4f�m�O'O�Z���0c:���X�.�3S��,�#��ƙ�J:�4 �����u�]���%5d��.@���a/����x��~{��Qb1d�4A&!Q�MT$����E�y̨KJ.e�T	��X}}JR*('��CJM@��*eN)u��8<��o��r�/G'��$�����F�� ۓ�����[ l��"���lk$�S�,/�]VY�o͵&��R��܀�$2�{<V����8}���
�h�.�Ӑ�� Ì4�XNKC9���RRc�o��K�/�Z�z��Q ._\]���|IE�Ȏ��1[�-�L���%��eb����aծ����fuyi�ɠ�Y��Hx�����#�����MM��E˗�a�`2��5k6m�-[SS6�*��S:wC��Y�j�0p}}= ��PW�v�z�~������lܻO$��5� p||�����I�
y"�H�
i]R����)�&O,��?� �B}�_�~夸�����Z��ǫ�?���ݷ�_�� �?�B�ї��}!y�+�>4��om�l\ք�;��}��W�{��������<� �m�>��}�dl>9r��\�p[����!�����;2�?���7�w6w���P >��=����:�S���s�@W� �w���( ��u��]����f�?�:���;�h<r�����P��Ξ�?y�2����O�� #�����[���y�
��{���^��1�f`�^~��K@�� )	� �k�Ǡ��=}�g����{�����@_�Z�5��N��J*���~`��)�Ovڭ�/�z�[o��m ��7_<�����	lˢ���� �ȧ��` 0��@�P��U}zg��w���vz��e����G�>�毯]��s/�~Lm������]z���W^��;ϝ<�W۰�QŅ�f��� wN�|PnU՜��y��yUU �g�FG�]>on0+�t���9U�s3��Y���<O0bv�̎ G�T�����̙��~���{mi��i�����fa pc��ɞ�
<x�P�#`�e+Vh���5� ��b`ޮ=;K�K�KK�oܸ��pS۱�G�i0wtuu��lj�wx��� �݇�oY_��:R֘�qɱ���?�>��
�\�SX�f��0�~�Nb�ۼY�@��+�BZ�� .Y�,]�X�.���7�Tjs+�2m��.���.��3���E���i]�@�)����"%N�� �'w��u?k��O�x����sp/��ƱbSc�M�qX2K�e	�,5�Y>[�������ilVR�T^
����/�������CVg�N��PBQ m��%��s���ª@���V�C�N�X�z)����L=� ��ѓ�T����2�\�b�3[
�e��Y
�#]aK�c�H����-A-�k�K�a��'�U�@��7 ��̡"&#F��R���SϦu��.!-����Ч[c�T�u�^��/�-�|&����K16&ӕ�|��h�$,V��-!���1�F����+� �)��z'���-�sp,�1���Wƣ�����,��Zh�>5�We���h�m�|oT��`��K��=j��KܲxU433���̑h9���s(�P��8   IDAT*�9̗%K�l��)�"�[)�A����QƎ�>��%m�C2֧ � U��-ulfR@'1��ī��J��	��i��L-����2�s��.���B;�w��_���F�- ��u�6)����i�%���ڮ)�#c�%à��wǐD��c;�Kc����
C�L�br��8��m��h���OO��>E�L#/dlB��6�*
�4 �8<3���D���44�aƠ/��+���p�i��cq�`B�O����ב�;U<�Pb���200��+��I�D�o�<-YL̿���D���Ԙ��ӱY�A@R(3�+3	��I\�U�V�s��MA���ŭ ��R���gpU���x�<�#��0����.t�7��L�5@]��0')̫b��<U*�;ͪ�2Я@�NW$t���֊6E�H�o9Jb�5�5F���Q�j �3�@/[5[dL;2�=:�T��Y�*O�ӕ�`c����lf�8[Xo���M�Gk/�\�[�*����f�M��x�0f�#L;���U
�*�_E��@�ёc ���	���9��vU��9�;ww�m���ز��ضξm�;�������>ڵm���������F �������ΟȮ��
��<�>h����ִ����+�\<?0t��n�2 ���ق�-�ZmPfJ�r4�LX1F`��#���;
-���zƉWN�yU>Z 0X �d@Wp�ɜi�$Z�H�"�I����o��Q9�V�`e�ȵHyV��� S�fl��/:X��R25G2�kOi(�
Lۥ"���E��.v��g��?'���#z�K�S	���du��. 7J�B)����S	w5A��/�0Z�j��L�_��I�!I�쒊�"��I�J�^���~� ,����y��| �ߴ C�p&�ל���"�^�Wk�L�]�:C�ui�m�0�@�H]b��E��$Ms��0-�Dc���Ul��y^Cؖ5wN9��e.[4�Z�q7Y�[�L��ɃEj ��Y��k��[6��X��f�򥋫�U�-/\�d��ښ���?���=�G����K��vGV~A�ʆ͛�6m޴c�ν��7n^�|m^aEfN^eU��K��֮^ML1��5���xSCD�No�o�Z[����D7�9x ��r�ĩ ��ؤ�dvr
���w!��+'�J�b�T,�<qR��$�d��kO?�
у7��?�Hu�n�(m|����~j������������?��K ���������|_|��P+�0!���a��/���S�C�'N��������t��w�'����!���ę3$���h�������$��Μ�&F���IZ��������S�h^hh��Y�2DsJ��<�����ЩQ�O�9����p�;z�����?t��X��{7v�w�t;����'��~��K�����������Y�{����������������#��׽��]�����n��λ �2�B!'ٞ�Z� LI�K�W�����~ �2m����vǅ��z�w�Ὓ����W�>�������0^��y��ʏ�8��O| ���;�]��w%��P��o�|zg�?�<��yB�I�3�ã/^y��7o���7����w�����?��e��?����O���S��~�W�y��w^