# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core import bbox_overlaps
from ..builder import HEADS
from .retina_head_trans import RetinaHead

EPS = 1e-12


@HEADS.register_module()
class FreeAnchorRetinaHead(RetinaHead):
    """FreeAnchor RetinaHead used in https://arxiv.org/abs/1909.02466.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 pre_anchor_topk=50,
                 bbox_thr=0.6,
                 gamma=2.0,
                 alpha=0.5,
                 **kwargs):
        super(FreeAnchorRetinaHead,
              self).__init__(num_classes, in_channels, stacked_convs, conv_cfg,
                             norm_cfg, **kwargs)

        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
        anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls.permute(0, 2, 3,
                        1).reshape(cls.size(0), -1, self.cls_out_channels)
            for cls in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)

        cls_prob = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_,
                bbox_preds_) in enumerate(
                    zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds)):

            with torch.no_grad():
                if len(gt_bboxes_) == 0:
                    image_box_prob = torch.zeros(
                        anchors_.size(0),
                        self.cls_out_channels).type_as(bbox_preds_)
                else:
                    # box_localization: a_{j}^{loc}, shape: [j, 4]
                    pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)

                    # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                    object_box_iou = bbox_overlaps(gt_bboxes_, pred_boxes)

                    # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                    t1 = self.bbox_thr
                    t2 = object_box_iou.max(
                        dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                    object_box_prob = ((object_box_iou - t1) /
                                       (t2 - t1)).clamp(
                                           min=0, max=1)

                    # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                    num_obj = gt_labels_.size(0)
                    indices = torch.stack([
                        torch.arange(num_obj).type_as(gt_labels_), gt_labels_
                    ],
                                          dim=0)
                    object_cls_box_prob = torch.sparse_coo_tensor(
                        indices, object_box_prob)

                    # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                    """
                    from "start" to "end" implement:
                    image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                     dim=0).t()

                    """
                    # start
                    box_cls_prob = torch.sparse.sum(
                        object_cls_box_prob, dim=0).to_dense()

                    indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                    if indices.numel() == 0:
                        image_box_prob = torch.zeros(
                            anchors_.size(0),
                            self.cls_out_channels).type_as(object_box_prob)
                    else:
                        nonzero_box_prob = torch.where(
                            (gt_labels_.unsqueeze(dim=-1) == indices[0]),
                            object_box_prob[:, indices[1]],
                            torch.tensor([
                                0
                            ]).type_as(object_box_prob)).max(dim=0).values

                        # upmap to shape [j, c]
                        image_box_prob = torch.sparse_coo_tensor(
                            indices.flip([0]),
                            nonzero_box_prob,
                            size=(anchors_.size(0),
                                  self.cls_out_channels)).to_dense()
                    # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                                 1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]
            matched_object_targets = self.bbox_coder.encode(
                matched_anchors,
                gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))
            loss_bbox = self.loss_bbox(
                bbox_preds_[matched],
                matched_object_targets,
                reduction_override='none').sum(-1)
            matched_box_prob = torch.exp(-loss_bbox)

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes_)
            positive_losses.append(
                self.positive_bag_loss(matched_cls_prob, matched_box_prob))
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(
            1, num_pos * self.pre_anchor_topk)

        # avoid the absence of gradients in regression subnet
        # when no ground-truth in a batch
        if num_pos == 0:
            positive_loss = bbox_preds.sum() * 0

        losses = {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss
        }
        return losses

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        """Compute positive bag loss.

        :math:`-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )`.

        :math:`P_{ij}^{cls}`: matched_cls_prob, classification probability of matched samples.

        :math:`P_{ij}^{loc}`: matched_box_prob, box probability of matched samples.

        Args:
            matched_cls_prob (Tensor): Classification probability of matched
                samples in shape (num_gt, pre_anchor_topk).
            matched_box_prob (Tensor): BBox probability of matched samples,
                in shape (num_gt, pre_anchor_topk).

        Returns:
            Tensor: Positive bag loss in shape (num_gt,).
        """  # noqa: E501, W605
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        # positive_bag_loss = -self.alpha * log(bag_prob)
        return self.alpha * F.binary_cross_entropy(
            bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob, box_prob):
        """Compute negative bag loss.

        :math:`FL((1 - P_{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}))`.

        :math:`P_{a_{j} \in A_{+}}`: Box_probability of matched samples.

        :math:`P_{j}^{bg}`: Classification probability of negative samples.

        Args:
            cls_prob (Tensor): Classification probability, in shape
                (num_img, num_anchors, num_classes).
            box_prob (Tensor): Box probability, in shape
                (num_img, num_anchors, num_classes).

        Returns:
            Tensor: Negative bag loss in shape (num_img, num_anchors, num_classes).
        """  # noqa: E501, W605
        prob = cls_prob * (1 - box_prob)
        # There are some cases when neg_prob = 0.
        # This will cause the neg_prob.log() to be inf without clamp.
        prob = prob.clamp(min=EPS, max=1 - EPS)
        negative_bag_loss = prob**self.gamma * F.binary_cross_entropy(
            prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ��IM���¹��ɉ���>�w���=[>��{p���sys}u��dq~zd�?�ɢ�m<�IL�g&��f����ب;0<��1�{m �r�:�B�mm(ͤ��f �@_ �VF�٥�{ث��i�ۗf��,�̒vR%p_���L�(�� pS]��`���B(萉��:�M>���{n�Ԡ$���t\ �6#Q%�`Я�%��^������@lh01��L��4�Uo^\�_3��e��Ɔ
���ڬ�٤%c�x؉��h6�%b.�wMFESc%�U�፧zÁ� �7�,v=���L�ɊZ<�����+(eU�{�U"���A�I��lfL��I�Q�~� p!�|AI!,�q%9 �2��͊�< ��������P��xn�/���ڭ�������]��U����:1^IN��^����bG7Ⱦ�2D_d�����wX�i��B��	�44�	hn$ �S�b�+���͕�W0�7���x��YG�i�@�pp+# �{�U��k{g���[;۫����z������n^�ze����k�+����?��_}1?;y�ء�� ��?z;<�|	��cs�T0�?�%�`xvq9�3P�i w��00���a	���D���wyc[$�<t>�Z4ir=	$\�D��9RXV:p-��fX�*��5�)�m�a=��7���������7�G2����@�W�|c�;o�V�B$���@�5�e]b Xc�謝F��f �A�NWg�qu6�0��N������z\}��I�0�L6�B�3;;:�D2|� a0�߅�����_Ńzk�Z��iB=�N�'�v-Ħ�0I��f3Xt �V�Xa��6�C��0����	��f���TJT*)D��8lv���R��>�����[���LMO������3S�C���h��Lذ�B����D�q��ku9�`����'�7����e��"<� �$ �&==5 �@�M�^����ݛ��o�Z�GÍ�u��ZJ'�Ɩ0yr�D��4���N������dq�\�D�nnk?y��D)�r��w�߿�����*�J��W�Y���.) ��F���q�"W0�R(�<><�F�6��`����^0���v{&3>3����uǝ����M��� G�]����X,��n��5ZCck�����U�-2�A"ױ��
B�����ݑ����ѩ�?"Si �d
6���������G�G@��k�n 0 ���]``�B
�F.asYJ�BcTw�y]BO�# sxLlRw��a�a��������������Vñ�z������vBYsE����DF���D `���&�@P���)!��s��	�+�F��x(���I���A�w9cQ$
����z�?�8`�B/�N78awf�֬����`t/�Q��A�E`x�^���X
 ��X��F�<O𮤪�J�dδ�l�([���,f����-�efNc���X	E�5U��3}3;;3{�p��1��_����T���gg#��7ɲ������\8��|�p8�@R� H8 �ba�/��@-��A3Bȣ�^Я��N�N4��6t�'�u��9��H���N�y�`u pL���L��S&ˠӕ�;����Rv �My�C��d25�j��R�$F��
[�H� S�V�/�(a 0Ee�RtƢ~-4� ���| ����b��a�B����,}*�У<2���4�MV';6�H�G)m�Wb���9*��:N���'����uv=@�;N�&�Z���l��X��,e,\�� /e� +Y|�r�_}d{���"}s� ��lNu�.R��0̓�ý7�2��1��iv��zD{b9��Hs����'��s�C��iq���z���F٠��Z��tR��e��^ ����X7�Zm�Po�5���kk�_�@�5��.�
���� q��al�"���V��1!s��B �(�H�ɢ�� �&� �ݛ�5�A���V�N� �V�5 #���_�k����?�;J���GLb�WS�?�]�\\20VN�8��Q{���F�K�ư�}� .��zS��@�U* �܉dO2�H{�=�D�/p&"}�^�I�'�WKkʅ��'�X�T�B]�G[�]�_����T�;}�t��UW����GC��5���H����%&����kf�7\�甶�2�E2�;�$����&��zlY�7ԣ��T��st�p�{ߏ^�XJ:�B���v0ӷ�5:�z9�T(],�t�)�+���������F�������B����/ {t8�-dl$�.��H��d��oY2�LD{Ї-�G�6�������>`h�p�sV���Brr4=3��������Sd����� �X�/���g��p���,.���,-���_�៎�>rݣ�۫�7o^9u�8��N%SHDЯH����
U�"K�|�7���H�x�D;������˗/>|�СCa=�����p��K�/]:��X���}>O4����������,�B><:��_}�vr�y_~���W��f�Z��,Fgkөc�{lo^?������Cu�����~�f���B>�N�|�lcM����Ͽ|���/��[� ���F��A�<vM��k����Ϟ<���z�����{ �d<5�k��䟌d�1�i���d@����m�:8
�K��z�:P��b᧦2�K��`�Mb3	 �����/�X!_>�E!��}0����h�"����D�<:�漙W>녯�� "��f�?��I�*��K� �t̘i�e:�+d-�c���1��l�C�|ƙ��ӮX������^��ק[�X�O������öD́}kbn��,�S�+  \">��P�E�^�!�V.��Ǣ��\]�y-�&   IDAT��F�I��FbkS�x?�G����t�������E���[�!��-������ �2����u�����`�!{-��"z��zHX����N<0f�N,�o\:|�Զ��Z��|!�kl��W�Htrف2BW'jksBh\�� ��]KSpյ�}�܂R쉍<܎zec,/^�vcoOm�g���q�4Z�����xJZ(�
�rɰE ��0Z��F/u0:�8]�[��[��L�PH\b��4!�#aq�lx�>>zH���x��o�>~����曗_=�����ӧ/^l���O�lmo�={���o^���
�;�6<���?��w��������0�D����2 p,�Ne�[;�C������.�X�+�"�N�Ј��_-D��@�]�~�݇O���2�� � P`��R�L�� q`�e�����.Wܽ��?���� z�N��J�046��aÖHkm#6�u4S����Bz�E���K�^���5Xi#�6�-g�)�E���<��%���|�P���	W,��,!�VMՁ���MO�������WZ5r�Š�*z)�_3��`Th4
0Z�-k#���Ǟ�&�g'�fƣ�@8���s��ز�ŵ� јM}}x�CPwO's�t>���k,~��dk��q�&�K�'R�X"�fς ���FB�L>�x�� �;���mdhzvjv^����IHxq~��lXXV�� ؛K�kkK��� ��ϟ�X�ήn�F�6�u�n��߭�s�|)O�(��k,��lw������trum	�V���c����>7�hj���^7$p�19���{����f�v��B��h���_Y+ē���I���O=z�8���FK"P'�'�.<M�T�しw���s�����KL��/��b�J��?�o��?������7_}��:<~�ͭ���B������Ex=s�ٹ�Y�6����͝�{�1	�<؜���n��,�T*��h0��!�F�Π��5�uz{�:��l�jxL�a[��Pl�'tt;	��`⭯m�5���N<\l���+��B�'u��p,<^��b�4�:)�3�va��=��k�$���X(��lQ�*�6	8��ʁ0�Vs���F���<�p!��F�^ʈ��Ph0;]v��_��2�tr`8*�걨/4\��`�v��d�M�O��������H,���g�?�0l���(�5�bsq��@�H/�w,���'|� `�ь8�=��2Hhf���	���K�-R��R#p!���2�>�6�7耾�5�0Vqp���q�k�t82�^�``<��Je��9>��fK%-E�#I5��C�!�MT�����-(дF��D��F�b`Y��Hd�X�4��;��C��{��呎��k���Q�h�ny��>��3ݶ3M�#D�����5e��#q�g+���1��H��~k���X��pS!t�&+�LV����Φ�3�Ks���Tӝt˝����j�y~�>NX����+�͒wh�_0��`����j�us�sg�<��ٯ��;6����bHY9ci5�;.�r�!!�%o�� ��F���$��/N��i9>�Y�4�'믧pՐXSs���ߌ�^�A��1����%�dJȼ=���N���Q���W0R�E�2��#ó$M^3��LBZ�@2Zʨ�+�h�2=�9"H+G�!��5�~'W�9*=E���;8�6� �A��2p��8�L)��K�WJ2��C��.�Q����XƆ�~ 0�wb4����o.��� @_�-1� �'�h��R0����ހ��c���l%�&�갓��Ζ����e�m�PN�n�?!}�e�����c����{�|�~�ŇMWv�/ܻq���sw*�L]:�_U�v�s���e܅�ugN�8y�X[��0<FY�ߔ��3��l�5ځx���bND_�����{4-^,�7���� ��^���%������{�.�0��v?�ԕMs���~���D�)�\��w-�����	��(`�p�
�����D�h�ا.�7ÇF��^�߉���v4�
5c���;����_0�G��*���6�� �����)�6ԽΝ��oeҩB>N�z��J�L�QAY�����
�@Ȥ3@�p���1lY&��C�R�x��+?��@�#O�>�`0L&}[[����N���J%��~�A��q@:�mP!��Ru��������0��Ұ�h[��?~��ފ |��q�R�����/��z��g�n\=w�4�k汘t2 ����řI ���/����@*|䣲���E�r� �gO���ųG;w7@�`���9�B ^\���ټ���������"�������� 2>�.�~�TJ�,V��
9���.�Ԥ�X�|Pk2j����J�}w��m�zr�R>0MLF�بP�u `Hn l�
�r4�5&G|�P� ɥ`���0���lO!��z�~�_
�v����&��>��·�bhT�P� �%Jg|85;3y�-N����bĺf��\�H����ﵶ5U���7X����]M�<!S���V�g;�����EALE�4�4b.��"w;���� \�0��{�3JͰh�-$�m�Z���H���vlp�� ��gO �s�y�T*�W��?��'旕�5���	�2��¦P9T00���1���@�G)� ��ݎ$�~�A�{ F ##��fa�i��M``8�0�%b�n��£e�J}��S�PGh���8 �r�L�uA�a�0�
�Z���]�o�;p�L*=y�����<{� ~�����W?�A F#����ޝ �Ջ?���gO����\&���7����G7�<T�@"Y �@R�4�
 �y����l���Đ��_plI�X�j��h����/�ojkn' ���\ �Bc �&_p/��m� ����n�{����A�t6�����VB}Kg]�����yx `��19Y�6���၁��G���Ƃ�'[H�9b�ÁP�0�㱅"�@��s�V�*;x�0> ��*oh�|�Q ��$�n�F+�+�� ����L�#�mW_��i�9�.���u�VVg�Ƴh���B"��].�����" �˞Jš����'�w�v�ysKmm]9�Jl���\'�I�?Ƭ*{�ŋW�?~v���;��o��/�Y������c+Kp���M���;�3�ks�K����hP�XT2��j�����F�V3�'�h2�"7� �l����
U&�/���Йl�j����`gieqfn�i2�-�����7[�K<|�ߨ
�# /O��n��95?�
ʔj����Ʀ���o������� ����o߿��g_}��uq`l��{#c��~?8�×�m(	6�����v]cC+�[c��B�L~u�����W�ZZ؜��3R�J�� �&����b�X&��[��a���l��!�w�ֶ�E����wW�?�Z\�+���6��T*��it��-���&�A����u��f�N��BΡ�j���+���t*���������ݖ����7�k����b��"�ń��(Mj%��hobwtt�٠����j���Պ:G����cGz]�>w����WH��h�7 Of��� �	Gb~�`*�HeҠӤ߇-����½y��T�o<v� ��Xb,E �� ����aBV[�N_l4���]Z= ����(��4�w0�@m�{k�z
���R"���G��b/h�@?j� ��v� �`�t04�\��w�lM`m�00jFA��6p�f }�^π�=��c ���N��vw�*]i�� �x��Kj )����(ףF`00藡7�~Y+���F#�`�Z�F��������P����&���w�w���N�����>|Jtc�G:�zX�=�cC�M��+#���3����L=�����l�n1ް�h�.e��ӍK�楡��B�|���:   IDATe(Xqg�}k��3C,x{��5Jx�Ȼ�mv�/�[V����ʄ� <��p�N2�
��x�uOé^ڱeo;�����z8�\�4�DqM �p;Q�V��
�@Ĕjlq;MHdc�	���
� w�t``(�:�oQȲV:�������-4Q+M� �ʐ������N���Γ5�