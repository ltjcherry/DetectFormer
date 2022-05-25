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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ¶‘IMì÷Â¹‚¹É‰ûÝß>¸wßïõ=[>úþ{p¼¼ºsys}uëâdq~zd¨?ÂÉ¢×m<°ILŒg&§†fçÆÐøØ¨;0<¼¹1»{m ìrô:±BÎmm(Í¤ü×f À@_ °VFõÙ¥¨{Ø«´ËiÕÛ—f®í,ØÌ’vR%p_ŸûLÉ(ö· pS]ÉØ`üâæB(è‰˜¤:§M>ö§{n§Ô $ºÌôt\ Ž6#Q%’`Ð¯Ó%·Ú^¿Áï³üŽÌ@lh01˜éLÂù4œUo^\˜_3›”e¸‚Æ†
øÀÿÚ¬½Ù¤%cžxØ‰þòh6©%b.ÎwMFESc%œUÃá§zÃ  ¬7©,v=¡¥æLÑÉŠZ<¾¦¸‹è+(eUå{œU"®¨¬AñI›çªlfL›àI¬Qñ“~Â p!î|AI!,†q%9 Ã2»‚ÍŠô< ¿ÿÁ×ö½ªPÉÉxnè/¾üüÚ­«ÄØØÐÀ]ÕõU¸š’ò:1^IN¿¨^½‡Áà½bG7È¾¯2D_dà€áÂ½wX¯iªýBšÚ	Ù44	hn$ ðSåb‡+ØÔÄÍ•¨W0è7×ãéxØÐYGè¨i¤@¿pp+# ï{óU¡¸k{gýÁÃ[;Û«¨¼»³zãÚê•ËÀ·n^ºzeõÆÕõkÛ+‹£°ý?¼ç§_}1?;yâØ¡Á øÇ?z;<Þ|	ƒÉcsíT0°?Œ%Þ`xvq9Ö3Pôi wÛÙ00è•‚a	ú¥ñDÅµ‡wyc[$×<t>¯Z4ir=	$\×DýÒ9RXV:p-¥øfXâ*›ð5ð)Àmša=ïÅ7ßûàÐÊÆÅÿö7ÿG2Ýú…‹@ßWß|cß;oÃVÑB$’ÚÚ@¿5e]b Xcî„è¬FÝæf €A¿NWg¢qu6§0™²NÍö¸¼ÊÚz\}žÜI¡0èL6ƒBí 3;;:ÉD2|å a0ð¡ß…ÝÅÿüÛ_ÅƒzkƒZÀÐiB=“Nõ'Ãv-Ä¦–0I‹Éf3Xt ªV¯Xaû6»C‡º0Àñ’ì	»Üf‰„¯TJT*)D«Ö8lvŸÏÊR«Õ>ŸïöíÝ[·®ÍLMOÂÎíò¥õí­‹3SàCÐÏh²êLØ°ÏB©ÒâðDãq·×ku9`’ ¯»'Ñ7îÏôe²Ó"<© ç$ Î&==5 û@ØMí^»ôÑÝÝ›·×oÜZ‹GÃõuøŠZJ'“Æ–0yr¾DÓÉ4“éäN¹“ÕÉìêdqé\¾D¥nnk?yö”D)½ríò½wîß¿½¶¶˜Ÿ*•JþöW¿Y˜ž­.) »ÌF·Åäq˜"W0äR(…<><ßF§6µ‘`ÿ›ìÃ^0üœ­v{&3>3½´¾†uÇŸúÎÍMÏÏÏ G¯]½ž‚“X,€¯nì„5ZCck¾šÐÞÉU¨-2¥A"×±¸â²
BÁ¹ó§ÏøÝ‘©±ù±Ñ©€?"Si Àd
6æÕõ¥ñÉ‡ËŠGÇG@¿ë›k n 0 äöû]``…B
äF.asYJµBcTw‰y]BOÀ# sxLlRw‡ÇaÁa±´´¸Ï¸²²¼ö…¸ØÕVÃ±Žz’ü£ìŽævBYsE¹¢²‹DF°´‹D `ƒ Ë&“@PùìÖ)!½«sÚÌ	¯+òFý®x(îíéI„Ã«Aïw9cQ$
õÚ€ûz¢?Ö8`³B/£N78awfŸÖ¬ï’ØÄò¡`t/€Qºí¶AúE`x^•ÎŒX
 þÿX»çFÓ<Oð®¤ªÌJªdÎ´ÓlË([²˜™,f²À¶˜-²efNc¦ÓÉX	EÝ5UÍÝ3}3;;3{·p±·1»÷_ÜïÕãT¹«ºgg#Öñ7É²ôŠÞ÷ùø¡\8‰ú|©p8‹@Rá H8 €ba /ã÷@-Œ·A3BÈ£½^Ð¯ßîNíN4Ýì6tŸ'êu•ú9ÿÀH¿°…N÷yÀ`u pL‡õÎL€S&Ë Ó•³;¡à‘ËRv úMy¼CÁÐd25Îjá ÌRˆ$Fà‘
[ßH¬ S”V /±(a 0EeýRtÆ¢~-4ƒ úóô| °ÝØð²§bäõaáBÇöùÞ,}*ÕÐ£<2¬˜Î4¬MV';6çHG)m¥Wbœ¤×9*Ž‡:N¿‰ž'´ƒœuv=@»;NÞ&ÌZò‘òÅlãæXÇî,e,\î× /eš +Y|Ür¡_}d{¬óÁ"}s’ ú…lNuì.Rž®0Ì“Ã½7×2¸Ý1Âãiv¾çzD{b9‰ÛHsÆëýä'õ•sÝCœÚiqË³Úzû°áFÙ éöZ”ôtRº•eöµ^ šòX7éZmŽPo¾5ÐÔékk‘_º@Æ5ë„â.¾
›ûŠ­ q•€al¬"ƒ÷ÂV‚1!s»ÉB ˜(è¦H°É¢¾ à&š ôÝ›ó5èA«û¢VâNž ÌVí5 #ô¢æ_økþÑäÏ?Œ;J™ì‹çGLbýWSþ?Ä]Ô\\20VNü8ÃÈQ{­ˆÅFàKËÆ°Õ}‹ .µýzS‰È@ÒU* ñ¦Ü‰dO2åH{ö=‹D¼/p&"}þ^»IÌ'ÕWKkÊ…·¯'¯X§TüB]õG[Ë]¸_›«þÚT÷;}õtµ¿UWýµ±òGCëï5¸ßÊH¿–¿à%&ü†ÕùkfÇ7\òç”¶™2ØE2â;í$Š°­Ù&àÆzlYŸ7Ô£óÙT™ˆst—pç“{ß^¸XJ:áB¢Ñv0Ó·×5:ã…z9¤T(],¥t²)¸+”‡²½ûÒÿá÷F£ÇÚÿÏ”ŸBºÔÂü/ {t8€-dl$Œ.¢·H××dÁÛoY2îLD{Ð‡-²G‚6¸ˆ½¹˜Š÷>`hpésVÉ‡ó±±Brr4=3‘ƒí¶¾ËàSdðÔÄÔó ÀXù/Œþ—g‡þpÍìþ,.ÎýÙ,-Íïå_áŸŽÞ>rÝ£‡Û««7o^9uê8à–N%SHDÐ¯HÀúª•
U·"K„|—7£‘»HxÀD;¤¶º¦üæ­Ë—/>|øÐ¡Ca=©ÿä®¹páì¥Kç/]:ÛØXÃáÐ}>O4²ÙÌ—³¿×ÓÞÚ,ÅB><:¼†_}õvrªy_~þäýW¯áf–Zš„,FgkÓ©c»{lo^?òòéë÷ŸCuçâåçÏ~ÆfÒÁÀB>›Níº|ñlcMåòÜôÏ¿|÷Åç/öÍ[à ¸¹¾FÂçAÄ<vMÅÍk—Îä¢Ïž<ØÝÙzöðÁ£{ àd<5¨kæàäŸŒd²1¨iÁÎd@¼ÌÃÅm±:8
õK¨özí:PŸÎbá§¦2ÛK‰`Mb3	 ‘€±ëƒ/ÎX!_>»E!éÚ}0ûù›ÍhÐ"äâå‚ÎDÐ<:Êæ¼™W>ë…¯€Ù "«çfž?¾›Iµ*ž€Kû étÌ˜iãe:ª+d-ÓcžÁ”1ÕælÃCÎ|Æ™°çÓ®XÈôëýýÚ^Ú×§[˜XœOà€èÓÇÃ¶DÌ}kbn™,æSá+  \">ÉåPÃEø^À!¾V.«¡Ç¢Ÿ›\]÷y-£&   IDAT¤ÎF³IõFbkS‘x?›GÁ°Ùõt­±µ¡ØÏÜEèÅÜ[”!šñ-­¸âàÔ †2Øµ…¶uóºŠÀþÃÍ`”!{-Ãí¸"zÛÐzHX¹ˆÒæN<0fàN,—o\:|ìÔ¶óðZ„ü|!ÞklŒúWïHtrÙ2BW'jksBh\ƒÏ ‘‚]KSpÕµÔ}“Ü‚Rì‰<ÜŽzec,/^vcoOmÅg´·èq©4ZùŒ±±ÁxJZ(Œ
¦rÉ°E £ü0Zõ¦F/u0:º8]€[¾’[ª€Lá“PH\b£¶4!˜#aq„lx¿>>zH¡”¾xýð›oß>~²ýðÑæ›—_=ÃüâÉöÓ§/^lï‘øó‡OŸlmoÏ={¶ù³o^ýõþ
¾;Ï6<”ýÛ?þþw¿ýöáƒ¯×Åá0ñD¨à‹Ä2 p,‘Neó[;»C£“•µø.ºX¡+ô"¹N Ðˆ” _-D¤Ò@š]Ž~ÿÝ‡OÞþÕ2èé ± P`ñåR¥LÜí q`‹e…‡š‚À.WÜ½ÿü?ÿó›™ zê³N¥‹J§046‹ÎaÃ–Hkm#6Èu4SÈêæBzúEÔìõKœ^¾ÝÍ5Xi#‘6Ï-g£)…E ²‰<©€%äÐø|†PÈ‰Ø	W,áˆÄ,!ÿVMÕŒMOý§ú¨”·×WZ5r›Å ×*z)ö_3›Æ`Th4
0ZÞ-k#´³‡ÇžÌ&¦g'¦fÆ£Ñ@8ìËåsóãØ²ÛÅµÇ Ñ˜M}}x©CPwO's¹t>Ÿ£k,~ýödkëœqŒ&‡Kœ'RÑX"ŠfÏ‚ ÔïôFBL>ŒxÃÑ Ø;Âémdhzvjv^¶ÑÙéIHxq~úÃlXXV–§ Ø›Kë‹kkKëëË ÍçÏŸå³XÞÎ®n¥F¢6ÉuÖn“ƒß­ë s©|)O¡(°k,‡ÛlwáÚñÍ©trum	öVŠ…ócÿø×³>7Ïhj˜ô™^7$pÃ19ìë÷{£©¤Ñf•v«¸B±Æhšœ_Y+Ä“ÉñÉI¸«ÇO=zò8º±µFK"P'§'œ.<M‘TÝã—wÀåòsª¶™ÉáKL®/–³b•JüÝ?ýoÿé?üßÿôÿû7_}ïï:<~£Í­ÐêúBáÅååùÅEx=sƒÙ¹…Yð6ðÖýÍÍ{÷1	ï<ØœžÎÁn£ÿ,«T*£Ñh0«±!ØFÎ Õê5°uz{”:à¨lÒjxLœa[àÀPlá'tt;	°…`â­¯mÆ5»ÄN<\lªÃþ+ÇBÆ'uÐÛp,<^Ååb¶4ý:)§3åva­¯=Ž¤k…$ü½¹X(lQã*à6	8¡àÊ0ÝVsÂçÏF¢Àà±<¶p!›ÉF°^Êˆ‘ùPh0;]v¥º_¯ñ¨»2Åtr`8*¸ê±¨/4\èõ`Ñv«ßdöMñOÔáêÑèÀÀùH,ï÷Ágî?Ý0l±¿‰(è5übsqåÒ@ßH/Ðw,“†Û'|ý `ÑŒ8=Áâ2Hhf¬ Ó	»š‹Kú-Rþ¾R#p!Îöû2Þ>ø6®7è€¾°5ü0VqpÁåêqÃk›t82½^¼Â…``<žšJe†ã9>‰ÅfK%-E®#I5¨¸C¨!ËMT•éú¥ª-(Ð´FºÎDÕèFÓb`Y³Hd•X­4·›;ìÇCòÄ{³¼å‘Ž¾k¹ŽµQÊh¬ny˜°>‹¿3Ý¶3MÙ#D¥—·ËÔ5e¶¶#qÕg+‘æÝ1òÎH×æ~k´ãÎXëòpS!tÓ&+›LV¯·®¶Î¦ê3ŽKs±ú•TÓtËÁ¶¼÷jÒy~µ>NXë‹Õ‡+ïÍ’wh _0ðÊ`ûïÖj¾us¤sgŒ<ªÎÙ¯­¥;6£ŒŒºbHY9ci5ø;.Œrë!!ü%oãé çôF ëñ$ûÙ/N»âi9>ÔYÉ4•'ë¯§pÕXSs¼¹ÅßŒ×^¾A¬ª1ò…‰º“%êdJÈ¼=ýâ™ÊN¶šÀQ¹°ÕW0RÂE 2¸“#Ã³$M^3•ßLBZ¨@2ZÊ¨+ï’hÉ2=Ú9"H+GÒ!ÀÃ5­~'WÂ«9*=E¬ìÈ;8â6¦ ÏA¶2pàŸ8‚L)åÏKÒWJ2ÑÿCö.°QÄññõXÆ†é~ 0èwb4‰úÄîo.õ€ @_´-1è Ž'œhŽ™R0ý‚µ¢Þ€ßÑcÓÉÅl%‹&ìê°“ðæÎ–™–‘‹eÛmŸPNìnÊ?!}ò„€e§íèýöcðÇ´{Ô|õ~ÃÅ‡MWvê/Ü»qæþ­sw*¯L]:Ù_Uí¸vsãõâeÜ…³ugNÝ8yŒX[©—0<FY¬ß”»3±žlÜ5ÚxÏþ¤bND_¨æÂñ¶{4-^,‰7“ìù× ¸ä^¸¸À%÷‚„ÑÌØ{Ø.¶0£×v?ƒÔ•Msõ£é¯~ÔÅÞDÔ)Ñ\êšw-±…ƒæß	úÌ(`àpÀ
‰†œá€…D›h¶Ø§.Ö7Ã‡F‡à^¤ß‰‘ô¿v4þ
5cÿ€‡;÷¢ò¿¦ø_0÷Gþ‹*žÿ×6ÿË †ºíüì)¶6Ô½ÎûoeÒ©B>NÏz­¶J…L§QAY¯Õ€»å
‘@È¤3@¿pú¢ó1lY&·øC¥Rñxü•+?ýô@ï‘#OŸ>Í`0L&}[[óåËçNŸþ”J%€~ƒAëîq@:ñmP!€ÇRuË…ìîîöÔô0Ôí Ò°ûh[¯×?~”ÔÞŠ |îÔqµRöäÑýÇ/ž¼z÷êg×n\=wæ4èkæ±˜t2 ¸öö­Å™I ðû×/¾ùúÍ@*|ä£²†êÛEýrÀ ä‹gOÀ™þÅ³G;w7@¿`à•¥9¥B ^\œÙÙÙ¼ëÞÎÚæÖòúÆ"–µÈêÊìÊò 2>–.¤~·TÂ•J©,V•Ú
9æÇÀ.»Ô¤ãX|Pk2jƒÀ¾ŒJ¼}wâí»møzrÍR>0MLF÷Ø¨PÉu `Hn lÐ
År4ˆ5&G|ÊPŸ É¥`àáŒœ0å³ÖÁlO!ïÊzÂ~è_
øvŒ†¦&¢€>ø’Â·ÕbhTÌPÀ ¶%Jg|85;3y€-N›²‹bÄºf„ú\¹H¼³µèïµ¶5U€áÐ7Xˆ¬Ü÷]M•<!S¡³¬VÀg;®¡¥”ÖEALE×4´4b.¶î"w;©Ä´Ã \ò0ê{÷3JÍ°hÌ-$©mÃZ†ÁŸHžÍøvlp‘Ä à“gO €sÃyàT*úWõõ?û’'æ—••5àê€	ô2»‹Â¦P9T00 ´´1š¾@†G)µ ïëÝŽ$Œ~ûA¿{ F ##Á¢faã½i·ˆM``8ì0•%bÀn§ËÂ£e¢J}¡÷SÛPGh—„–8 ƒr¡LæuAÐa0º
°Zƒ±á]«o®;p¨L*=y¾óõÏß<{¾ ~ÿù“·¯W?ÚA F#¡©°îÞ ýÕ‹?üþÛgOœøôã\&ù÷÷7ßýÕûG7…<Tâ@"Y ”@Ré4°
 ¼yïþèäl®µƒÄ© _plI¿X”j®“hëí¿÷è©/¯ojkn' ú¢©\ °Bc £&_p/ Ñmá ðÇÇÎnÝ{ºà€A¿t6è¡²˜VB}Kg]·áìï¶yx `‹‹19Yæ6Ð×áá±­Gˆ¨‡Æ‚±'[H€9b™ÃPà0Èã±…"–@Èàs¯VÜ*;x 0> Îà*oh¥|³Q Öë$ÊnžF+‘+° À™¾äL‹#ämW_Ýi±9Ì.—Íëu¥VVgÇÆ³hÒõé™B"ôù].—Ãçëñ" ÃËžJÅ¡üêÍÓ'ÏwívüysKmm]9™JlÀÕò\'üI ?Æ¬*{ýÅ‹WïŸ?~vïîý;£ãoŸ‹/ÂYí–áÑ±Èc+KpúÀÆMŠÀ‰;«3ëksËK“Á€ÇhP²XT2¹ÓjµŽƒÓÇFìV3—'Ôh2­"7Ø Àl©Š§Ð
U& /ŠÁâÐ™lj†¼ÿ`gieqfnÚi2¬-Ìýû¿û7[‹K<|«ß¨
ô# /OÔÚn·×95?×
Ê”j‡§ÒëÆ¦§¶îoÃ÷ÚåñÀ‡ üæÝçoß¿ùúg_}þöuq`léÝ{#cÃý~?8™Ã—™m(	6¯»ª¶ýv]cC+¾[cðúBñL~uëÞÏù›Wï¿ZZØœ›¹3R˜J§† À&»ÇæÆb²X&“¼[‰…a·ïîl€±!Çw×Ö¶±E€±¾ÐwWï?ÜZ\ž+Œä½î6“®T*‡Íitºá-¶š­&¬AØåèéu™ìfNßBÎ¡Ójªª«+«ÀÀt*­µ¹¥±¾¡©§Ý–¦ÆÛå7ëk«áÄÍbÐà"ŸÅ„³œ(Mj%¯Ïhobwtt³Ù ß€ûÔjÄà˜ÕŠ:G¶¨Ó´cGz]Ñ>w°ª”ØWH˜©hõ7 OfãÑ³ œ	Gb~ÿ`*•HeÒ Ó¤ß‡-·ÛçýÂ½yôÆTŸo<v« –‰Xb,E ö— œîó‚ÀaBV[¯N_l4ö€œ]Z= ñö€ó±(˜¶4îw0•@m¿{kõz
ÉÄìR"ëëÃG¡àb/hÇ@?j£ ûívà Þ`¬t04¦\ž·wÀlM`m¿00jFA€³6pÜf }Ó^Ï€Ç=ð‡c àÑä N¤ývw›*]iì’é ÀxºKj )ÌÀ©(×£F`00è—¡7ƒ~Y+èúåF#É`èŠZªFãñæù¡©¿ÜP€û§&’»w¨w—ºîN’ÀÀò‚…>|JtcÓG:ÆzX =ŸcCîMÁÀ+#Íëí3éºïÅÙL=––‚÷Æl´n1Þ°”hÖ.ešæÓK¹æ¥¡ÖåBÛ|®Éì:   IDATe(Xqg´}kŠ¸3C,x{Œ¸5Jx±È»“mv—/Å[Vü¤¨äÊ„¡ <À¬páN2ª
¬šx×uOÃ©^Ú±eo;ðˆ¤ÖÛz8Û\ý4ÕDqM àp;QõVû­
“@Ä”jlq;MHdcã	Üî–
è wñt``(:ÐoQÈ²V:¸—ßÆµÒÅ-4Q+MŠ ÜÊÅºÒîâ¹âN¾ôÛÎ“5³