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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             43Iö¢_” »“”TUc MÜHj13·á,odfafÙÕY €á›{w›Ü„»·ªÊ^d¦$ÒªfªÎ. Ý-‰dw“03Ih¹{U‘Ä‹ˆ¨ª9gØ%U IU}}zÞ¶í|>Q•k¦$3Û÷}láî$ŸžžžŸ.k•{zºd¦ù ÙÎ0oˆBDkþÓúïß¿ÿùÏ~ÚöËãÓ·ÿ÷b£ea™y}¾Ü6ÒïïïO§SwÇÑÝ îîîÜýùéòñãG¶mËUÝ=Æ àî]Uwïî‚N§€*×Z—ËÅÝÕ¸9ŸÏ•¹2IFÄ¾ï1àÃ‡ÇºI’«rÛétR–¤9§9÷}õêÕãùùùz½6H7ëX‘ì¬aãáÍkÂçº^.sN¸(5ZAËY~÷æ>¶qº»#ÍÎ²ðçãê•y<_®3k-3CöÍù|÷úíkÛ½?||zz‚ÑÀƒö`º{fS¸	Ò0—±ú»[ìÛoüô'_}óÍÓÓ
U:Ž#Õ™Ù¸éóã8ÐZkÏIŸ>}º®¹jŽˆ^	·ÌÜmYUMÌ\Ns
€ª#@fJ"]ÒØ¶‡Ww_ý5ZÇÌ£3Ì«df™ùüüÜ«RÕî^U ªÀÝd¦û€UwKç«RÕïß¼5òùry>®€U•»›»:»±ïûœ3ˆ‚Æ0f¦ªß¾~õÍ7ß\.O ªôÝ÷/—K– Œ1ºŠ¤€®ÊÌw¯_ýä×ÿË> ÈLÞ×y+‹P•æœÂUµï;ê& ²n2«{Û¶‘½àÖÝcÛæuMR¸Kênw73©ÌlŒ‘™U™Y«òîî® %EÄåX’†y¸xýæM¸Èšóº²ê˜´ÆÝð¦ª¶m“ÔÎã8<¶~af™	´ÄpJr/$ °*#BF-®u˜…23†EÄ¾ï$ìcTUw›™$šÜ†™e`R™YfÒÀõ2/—ƒdwgÎ&‚æf ¾ùñø£$#ª·m“ÔY¼ZR±mÛZ‹¤»›Ywg•YˆM²³2³ª¶}t13Ç$»{x 3ÉuÌï¾ûÎ„ÌŒˆ2ü'ÝM·ÎŠ —TUÝ½m›«*hÝ0Ãêr÷†rµ©ÜÝÌÔ‰–»”½ºjs¯*3#­³ÜÝÌ$È\[Ä8w¯ß˜ˆÇÇÇo¿û0/×Ì&yGw´mÛù¼¹‡™u·	ð É¶v°*DD"ërÏÏG„mç°a¥dUÛvê¬\MÒ=æ<,¼	3“Ô+8­Ö‚ã|Þ30·1çÜ÷ýx¾Ê°m[+suUKênIîõbŒQRfº;…îF53£™ŒsNwÏL’cŒªr÷î6cÐS½m£ªHäUµÖ²$1šºAµ{ä\$ÍPwg¦’fVU 2kŒæí|÷îÝØ¶îÎ:P±jZŒ¢UZÃ|Í)i¸Í9•5Æ0òùrÙ¶MRUÍÊë<DtVwÏ9;ëÕÝ=€±9od ¹33Ýýt:].Ïö…“|óæÍOû×ÿÇÿþŸõîaw?]þÝøËo¿û|T&¢ÇŸ>>ÿÙŸýùÓãZ3W>oÃ–ÊÝìn wÛV—$_•‰J=z´p’h­ã²mÃ»ÛÌV¯ya¾ÖênÍ¬ªH•zøÖÝf¶ŽYU»…»‡Y6¤r÷9'Ý®,3{>®§ÓÉÁy=Îç³Sº’lltSó¦»#âX“‚…;$XD€³@VÉý|Bµ^ðæÿ7´m›»Çáî ‡ü¸\íÃ·R¢åî†¦@7U“Î–Tî£ªR=ÌoŽã ¹m1ç”´ï;nšBF3ëlÁÆø¢»ÊŒAw²2Ý½%»iÝ¸;Ñ½Ò@£xãfð•YêÍcÀ:ƒ»5·¨1¾½ßeÚi¬Â|¯_½Í¶Ÿÿõ/.O³ú )Z‰ƒfžª™y5B„*GDn½ÃïN'~~üQŒqk2|
ßç¼Ò½!À(24	3ëîk¨!žW¾c<tmÆè¶Òô˜Ã©õ7u\‰{ÛO`Âœ-#—ìA9º    IDAT€³zy—Ew™^˜™ªÜ½ºu-
¨ªžg]º“ï{©n' ©MdŽ†y#hliD Ê`$“ÊÌ Ý ÕÝî^¥v5-%¡af]heDT	3+õ’†yU›ZÝ=Æ¨*@7îýÂÝÍ¬ªøBRfJdÿ@ÝxÞRw;·Ìéè&Éƒ>|?ŽK•tãÆC$õ¢ùÿÑ'Ë’eÇ•`÷VÕs¯Ùó6ˆ@€ Ñdƒ¬IQ¤þŠI‘ÌAÖ·rRBa²E ZwÏÌî9ªºËÜ˜µîÜÍA ÃÃÌœ„™»ïçÓ‘+%ÉÝ«
Àœ“¤ŒÝug%’’º;"Ü]RfJ0†µV	tRèºº¢»Kµfº»¤ˆØö±m‘™Ý¸]Ëåâã|¹>·cfîOgRsN·ó“ûÓŸþôÍ7ßüå/ž3ûX~øqÖ\k¹ûZ³»ÏûgfUµÖ’û¾“4³ëõZU $u)"2“w†ïUX7
k¯ªõ«HcÌ9«*"ÆÝÛ$?|øp¹M€ý`fû¾»;ºÌlÎ>Þ¿{+bÎy»Ý²eÙ‹µÒAÆxýú‰ôµVw¯µ”UPV­µÌ ç·¯csßOÃOk!ûv9nfVsÝÍ9k¥DóXã´}ùË/`,èùÃó]ƒ€¶Rc3sCwK¢DU1ÌlÍI³‚ÌbìÛ_ÿñw_ó«ëõÚy;ªÔÝ+³rº;+[Ìudf×Z·ãã‡çëºŠêR	.šYÐÔÝÉÌ¼æÒil•rCD8±Um>öÅû·ïß¿¯¹.—‹è$%æÃår1ÃZKÉ™­1†™I”TUºcK23InÃü¶¦$ZAkóˆ€ÔŸ%Œ9Æ€™ªÍìW¿üêt™éî™ýñùùr¹Þn7Z˜TÝM3Hfxûöí¯~õ«Ëó§µÖqcŒˆ¸^¯ÇÊ——N§§O×ËîQ„¤î&ifê&if™YUîî„ñ:¶Ø[
UvDTH#Ý=6Wõív-‰¤Ý0º{ªÕì•FnñæÍ›î&pÌY½Ö1oóØbìû^UÝíîý0Æ8:«
äZef’:‹$Ø`­efÛ¶‰ÈL3[kIÚÆiÎ))"Vkw;¶ØFb˜“’ ØCwÐCÄèn=ÐMÒ§/·Û­ªÂ†P™	 "Hþú7ß|õ›_£ZF§I‚Ô u·»s’t³ªŠ5Ecd•:;ëzÜ¶á´ i (ìn3«ªîÎÌ?þDº3–Z’»K"YU×CD4æ"ÖZ!—Q ÝG©×:àFÊZî®j©N§SÞÕÌÎáá`f4¸TcI€H¾~:}ñÅÇqÌ9?~|ž·#³VÞõ¶mk­Ûí&#€}OOç1¶R«z?=­µºÛƒ $e.wo‚Ù$L•™æªM›Á+…Ò¬ ÜcÕ2³T³EQj3sw°%µØYc}3V•$>Hªj©l`Î5F h@­òVªÎ-¶F«a#2»j03 RK"Ý-‰n†;Ì%u·$ «Ò@ tKÕ¸ƒ7T·9+Ý}ÛF·Žãfæ†IUEÆÓéôþý{!‰­Y)ÀÂ³­‚†;)sªÚ°1{Øm«³ZfQêÛåZU$‡²ünØÓÓÓ¼¤·D´Nc»­á>FD|ñÅû¿ý¿ÿöOüÝ)b?½ú×ÿîÿåÏ/ÇÊ®-Â&¼\æÿúÇùñç«ªus·µ³0³ÌI™PÀÂÑšh’™YUfV«‘½$Yœ$­š­Ìœ*Ì9ÝdD Èœ0nã	*’ j®Z°%ÁÌÝ3sŒÁ-æåE€í»ªQ=ÆÞ¹Z)`Dì§SU±Ã¶F‘Œ°ënU› 3«ªˆXkÁˆÖ¶m€uMAðïþÇßhv7É@¨;{nf$»›äî¦nIfÖ’Á³¦š1Æ îÚÌ@§i­i°»Ì5ÜIZI%7zDvYso••s·MÒ Š¶$3“ä4rÛ¶I‚Œu«fNÒ—²/öÓuû©ùO?ÿ¤ÓfÛ>‘¶íoÞ1Æþýw?}üpËbP³»ÝÝ(aQ©ž€)Ø¶mçýixÙZïú­ÆëyŒ[çaú©VÅ(CÑÌš»­IŠ(k¦ûÝO¹~Áq*ž¥€%ùO[ücß¾Íƒô/÷ó™´ÎáZR=Ý& #Í&>#i VZ8Â%ÕÊ03È…AH:Öºu_‰rë ¹EfZkÅCUõA@È,<g)`¥ìnU˜™÷ƒ»kÚp "Ô¸ëªˆ¨*ÆÏz¥ŒÝ=Æ À–Ô¤™I$û³:N’2SÉ1FÎ”Ú= I¸«*³h€ˆè3c¸™udw{p@Ò¶m]ÈÌî`f™iäãš7Iþ`RwK2óZkßw îÎÏlUŽ1ö}7°ªf®Ø74TUgY	ÀZ‹dCaž™ »»ªP0q½Übøã8wïî5Sè.©À~Þ·m3k4os]$õòr½="¶¥¬*‰{üõ_ÿæÿü¯ÿõþá>~xþâ‹/~þî§¹n¤ò³åä¶mûÝØZkÍ9×Zû¾o Žãèn’’ Þ­µHšéÝI:X¹ÊÃ ó ²«
Àœ³»Œ1Î§Sl±mÐ—Û¼\®Çqà¡»ŸžÎwèFËGTÕ¾ïqy¹Îu@ÜÎ§ª1+Ùr÷áá#ÐêFw_®W=ôÃZ+ÜÍì˜óôæÕù|>_ùi›/WÀHV‰ÔZ+³çœ9WfŒðÓ>Þ¾3¶mùòrûñçÝ½»#bÛ‚äcÎéîêŽMZ©*3§›š™Y7žöÓoÿøû_üúW—ÛÝš™Ù·:$eÎî¦PUÖ¢¬;o/—ï~ü©×4‡ŒëºHÆCföCU%2«F„ƒÎÛ.cCfV„™íûþÅû÷›Çº›%£$ÀÖZ’.—‹¤µ–C2V¯ê62|»ÝnÝíî D8MRvÞµHJU€b›Ùðæ™	  wïnwïîo¾þj›T ²õéÓ§Û\Ï·Ã,",ç€ÛA€¿üú—{ùôÑèBC¤u•ºt½^Wé Í@fš™»w7I Ý)"2gD ¨ªìrFd4‘"dw»=Öš—ãÈÌ§ýT+E‘&‰p7”š‚Y¸{Œû&â¸]²ÄµHFDUÁˆ I0V•í¬Tv‘\k¡Àdû¾gÝ0çÜN;II Žãpw3ë.€}W ùæÍ›±9Z x×‚ÓÌ$èîa!‰dwîžŸõq·ã¨Õf±òjfv>Ÿ}Ä›÷oùÕ× Ôí62“¤ Pj´ HªZ4PtÝÀÌHkº»™­µÂÜÌV%ŒÝ½Åp÷ZiBAÝíî™ùÝwß‘4áŽdf®®1Œ ²‹`$»ÛÌ$‘4†îP¹»¤ì²‡9§»¬i[f@Rk`xfžÆnfsaîäÌt÷ýnŒ9çóóómNƒBwÏ™UõôôtÇår±Ž¶y:^½zPÕUE7ÈÀ–f ÖZfæîUÕ™«zÛ†o63Ky>mëVîûÎfwg¦™1²Õh
‘sŠŸEDfŽ12³»Ã …UécË\ ""3Ý½»)EDIs.»ë,ºU•Ó:ÓÝù `U™Yf¹[wWÕc­efîž™ö ÉD3ëîµI ’fewÇsÎˆcs€ã8ÌÌÝo·‰mÛº»>kQÝmæ ÎçÓÛ/¿z¨nÀè¦ffÂ@w;X9×1í.¼»%Q8Ž£ªf€Ijèv¹6´Y :"¶1ÌŒl€f°Ì43Ò³g8=¶}ßÿê¯~ó·û½}sVj;=ýóŸ¿ýáÇkB¡ÝmÎÛZúÇÿ÷?~¸Ì9©¦©Ô”`R7Í tÕZë¼íf¯‚,ëÖ4‘¬ÛÍGþ|½û¸Þ6wŸsÛÀÃ¼§§óu^Í‚dUm™	Ài"œÖÐõåòôô£¤Yiá¼ØZkmÃI™ÅÝ\w-šÙ¾ïF.¶$’B^½ŒI¨3Ã]wˆy$ÿþÿù¤“ª^ ‚ÖuKefAs×yóadkD8 7€ ÃMÙif0JrZ³ò(IÝ£™#CD–`¢L’QÞpr»;È"}¸	óz0ÌGPÌL3¹Öí€Ý½Á¾h}mÛ>â%ø½á_o—«ø´·G…öWOoß~ùéãíÓO—[Ó,ÄVa­t@aMëp‘-	æñzß¢>=]ú+ßýò"áRu˜ŽÀµ\¦…ni§oàfìE^³
2‘Ð€G×»o²Ï ÍÄáñ^ÿ+[÷þv?Ù¨rÒit[À4Ýª-<©%¸; ¶TÛ¾ªèÀÕl™z'í¶æj]Àg¶<Ò›“¬Êîv÷mÛŽãd#$uk˜wwUùÃÑ«»Ã\Ùª63w'¹*%1ˆÌ½Š$ ’Nk€»ç\ñÁ@ fÖÝsÎ!ÁŒU "Ðàî"T-@¥Ì I=t÷£ªÆ’ü¡ªHºå’Zbu$ef2¬•fF6`A#ZwîN@v›YU™™Œ’öNåîf66¿;Ž#Ì%­Y H6ÔYxèîªr÷9—¹Üîºçœ]ªRå,cÄ«×{ŒÝ,n·ÛóåèîÌüøé“Ä³’H*åîpûý÷§?ýéŸþéŸÖ1ß¼y÷çû×y\«ÊÝ¯/Ï$OÛ8¿zÍ‡Ì<Ž£»ÇÛ¶E„™eæq$%¹»Ýía¹TUsNIÝíÎóùnsÎc®Ì4³ª:ŽcßwI žžžbø¶…™]nóãÇOÇq˜™$3cœN'ˆ"¹ï{D¼\¯ªÊU2FDvÝ¹{g‘¾ï; s¿¼¼ÜŽƒdUAên¶äf-Ž€Ûë×¯}Æ¨^]v'ÉÝó³¾¼\ãp#îZ§óöúí«mß;õÃ?­µèf ¤F½:?u·Pj‚-Õfl"È9—»%îlxl¾ýúw¿ýí~»ÝægYUs-¡$u'««Ê!–wgUýøãÇív¹ Ü)‰ ÌÒœ& f…ˆØ¶ä­Ö~Þ»û‹/¿|:=	=o³ŽŠ}#y»Íî^kU•™IªZsNÝ¡Fì/×IînfUef™ ú3ì1$X½ìÀÓÓ‰$€ U w—”Ôð0á«¯¹…« Ô1óºæóår]Iôil9 wïn¿þõ¯?|øp\oÃ€ºAº»Œl½ÜŽª’c¬* îÞÝî^U hjŠÎŠŸ·åîxènIîÝ™,ß}ùEf^®×5§2†RìÛÌ%ét:AÖJ’2°,"àñóÏ?¯Ê8ŸÏ¯N{R$U]UcŒ~8Žc­E7Þ…ß®S¢t3]+ÂÍ¼Ô’H¹Ì,3Œˆî&)˜§±m{ ï´y@jƒ$’ ú!bðÁÝ%e¦$ Ý03šÑÔÝŽ1ÖZoÞ½}óÅ{I™9ÆŽIwo¨ÔúO•f&‰Ÿ9ÉuLÑ0:ãv»1º[Ä2³0—d¥&ºÛÌ2ó‡~ˆˆÌd@w3\„¤ªŠ Ýˆˆª’ÀÝHª”ˆ1Ffv·Y˜ARwFÑªÆ[$[™sÅ ÆîH®µÜÀpÿôñù¶æ¾ïtîU}½^ñ™1æœkÛÊã´íÞÐð¨jÀÄ®*ÍXU$ÝÝÌV•KâglÈÀ†Ü¶­ªLÖÝf†‘UÕÝc}†;©Ø™$Í@vKrwºuI 	»sƒVÕB;L’oÃJUé¤»w7 ’·9ÍLcŒÌ$	@’™u·$3`f¤­µHVªm„¤µ–»Ó©êR‡9iÝM²ª$‘nÙ]U™IR¤™u÷ù|~ýî-HIÔ\•›/·¹m[w`+h·yt·HJ5gv÷œsUªš¤ÄîVufžN§µÐOûilŽ–»w·$’Ýeá$÷ÏNøÃþæoþ&s¹{ìûüùû—Ë>,|^^:õÏÿòç¾û^âZ‡„„ªæ² 2s'ÙGIµïçRÎJßFWõ]ARØèœU+³3Ó·±Ö!i˜K”*"º»ªÆë˜®°9ç¶m’†GCµrÛ6©NãtÜ.ykß\4™æª™Ç¾mf¦jœ¶€T»{Ä6ç4ûùäî (DDU ÉÝTšà›¡»ÝŒdÐøwÿóï!	š[M "˜íîFZµQA­m;ÍJ ëNa€º©F!fÝŽ$§}læÑm‚u1$­µ†‡µÜÍÍ¬”Y±E$»”µ ú’Î\«*ºÝQáU¯_Úövcÿö­Ö÷3o”í'v†?=½¦oÏŸn·„€ølËìãz£ªç×¶#¶0ÐÌœôZ>¯§O—_—Ió¶ªËÆó6¤
À@œŒN¦á¢,çªvbˆÐ¯ýtZó¬~[¯ÚÏÿlým!?g÷]tµ¤nøˆ"nìY%zY7H3ƒCÕ4§YxwP.oçÖt‰´£òø¹3·Maíæ™ IM(H…ì:ùVUfÖÝ‘™MÜ9ˆÖI3ëïD¬µŒl©š¤»K‚ñ®ªR h ª
­Ø€ZiH‘R»»¤nI R¤DUóÎ´¹”@’™u·$w'YUQUvÇ ©nF‰ Yƒ HJ-)"º[’Ó™îŒÇqà3ÞÉ­«†9ZÞÝp:Û¶©zß÷µÖp·eÛpšsR‹ôÎºa 
ëá8ŽÎ^k¤óÍùÕ«7O¢Ýn·O/·¬Êµ^.4Û_¿^ÇÌÌ·á[Œo~ûÍïþð‡©õüÓ'VÿðÝ÷×OÍ,3¯×k_¿zs~:‘\kÇ1çŒ‡}ßÝÝÌ2sÎ[7Æ’ºdf’2ór¹u·$R¯ß¼:N•=×1çìRfÎ9«êt:™™»1Nçýõë§Ø·?úéÃ‡œKÉˆÚþ@ÒÝ¿úê+I×ëõùÓKW#€îvwUwãt:mã$Ôåz™éAÈæœ$Ãi-½ùò=Ý¶ØoÇaö@²ªŽc=??Ï,ƒ†G­<·÷_¾L°çŸž/W3SC½ l{ 5†gËÀª5Æ ÉÌæœ™9Ì>b7Ùïþø_þðüá§çu›³ºª²J(Iµff jIÛðÛíöÓw?}üù“ÄmÛ2'Éý4º›$ ©2³ªò®ËÝ·îN2"ÊäÃNçWïÞ¾E#3o×YÂ¾mÛ®/·ªZ’ÌìX·ª
óî^káÁÌªÊÌ T•»“ìnÞy˜YËÌ\-77yÚîªIÚÿ@ªPÚ¶í¯~ó«µÖðt=nŸ^®—ÛäšåA¬*t·|D˜ûˆœ‹­ª‚dî±owQêy[àg///U’Hº;€î63¢d×cÎw7p¢IBª*3Œ±·”™ãõë§cÍ———Ì¬ªóiœc`fZfA‹ðÛu}x¾>??›ÙFûîé¼‡[ (@w¨ªµVfÆ6ÜÆív“q­4wŠðîgCîžjÂ3;ëxºªÀ„}ßOûü,Ü—™õI3€$€ª23I¼s¢AgU9ÍÉì>¿}óæíÛÊÌ*÷±Åènïîì‚ÌhÝñÎ­³Hâ¡¡îæÃ¼^R˜h "†»ª;«»¸{wÿðÃ HV•É2Ó·±*Ý½»ÇÝ½Ör÷–€ÿ[ßÑºÃ*U½Ü]ISÆUYUÌŽ}c+‘¨:ŸÏÝ-”É#¢ªx'‰ÌcU§$wI©»q'J"	`V6tÚöÌl´Óî$ED·î u·û¾WI Ý’²Ýmf[€Ê–DÒœîž™3“d©FÒÌú!ÌÐ‚ÑÌH®ªîvfÝ€k-’€ªr° P•˜YwKrZCHrwIUEÒÌHf¦™­µü@w“ÌL’f@RCUef$2‹Â»w·$ ’ yÐ™YÕfæn&úiómDDw¯µZ”d>lµ²»lUÕÝ™©N
žs5jUç<æLIîƒä\««N§Sw_¯×}ÄæNt…¬5ö]R©„ÅÓëWû¾ýþ÷xzóº»~ñþKÂ¿ûñø&qf^.—Ÿüñ/ßþpGÎrÂ\™	6ÈÊ@´™í®fXë Ù4˜2ûÎÌ ×©¬Ì	˜2è1†»Ï93ÓÌ"¢»–]c5Æˆˆî>ŽEêt:hÅîžsÒ-Õt+eÜ¬³$9UëèŒØ T•Ûð±oÖm‹VÊÌÂ|UV#Ü£GUµõà0o
üoÿó¿Kê•s¶ Íd¢Ë@VË’HºF™™îÜ$±
 ¤"Ú}Þ	2ßŒ§pŸ58b »II‘-7šYgUulafÙ%cw«¬.óT—P„»[xw«p2 óù+Ûˆþäü~ðÛ9Ÿ³Ó¼OC§ó+ûJ5íÝû/ß¾}+éväÏùñçO?£RèÓÎý¼“r³ž·OõãÇoVÃÑ×—‚Ú|Õ|îºxF2t6é®.bì§kçw¹6³ÜLgqÌÞ¥ÍM³—ðóiü«æµ±sœz½òˆFÈî
]$‡]*”p›“nŒ1Œ*Á ²Êº1ÂÏm=×n1‰¨ï+çfr!‰É£3ÌE„y©ïœ.µÃÌ˜Y·Ì4!Ì°ef2f&ŒÍÊÉª
#V•’`¼«*3ƒ[w“ , $%1;Uc3#	cw¨;ÕM˜ïdÙÕÝ[ìamfU/’îN!»ÌÌÝ»@$h¥»£UjwçƒªÝÝŒ¤63|Fw“ÔÝÌ†Ì”ÀðÌ4aŒa ŒxÈÌ0¿“Ý’bÛ€yl!#iÝÕÃ}Ø†N§ÁØÝ’>|øðòr­*
ªÎÌ§§§·ïžæÒÊZ­ªz~~^kÑìz½¾yóîÈÕæ4¿ùæë¿þãnÇåÇ~ÎÛ5×:.×ÎªcÞŽãý»·ûùdf ªêùùÙÝìû$ÕÝj
eŒne¦² ëîËå6†GØ~ÚÌlÛ°Ûí6ç¬ìªš•øOÕöð«o¾~ýö-Ø/Ï×OŸžã `fîÞ]zp2îÌ÷óé|zúö/ÎUû¶-5ŒÝM²ªÂÀ¶m‚IZkÝn·#Ww»{U‘DuW1¶ý<ž67ÆZ+³Ÿžžô`f9ëry¾^¯	£ªOûx÷Å[3ÄØ?|øôéÓ‹$ RmÃºƒQ3nšI²ÍÖZ><¶cÿëÿò‡o~ýë—gUgvUu÷ÊìZÝ=ç”D•úŒÏ?}xþø©»O1V¥|Uå]E
@UÍ\G®}ß‡9Zæ"bß^¿~M3HónUD¸;îšk­Ì¼Ýn±yw»®J’çýt¹ÜÜ’æœ’"B’»w7IÝÑ(H‚‘™û¾KŠ7øXUÛ¶õ€Þ¥§7¯þê›ojM’—ÛµaUýÓO®×ëØ6u›:Õ$Í¬ª¶m»ÝnÊ"ÙÄ«W¯Æf¶ïûùÕÓ£™ùí·ßæZ$Ý½ªÌŒ$€ªŠˆª "ÐeôˆmvAŸ±ËÌÞíáañÝ·ßïçSfÏ9U­\¯_·mt0 a¼`@·ž_ŽŸaÑÝèüú«/^öÌtw $\$×í(ˆDf‰ ,«$™YDd¶ºÛ$»Ûb[k ©»Ý´µÖqOOO•2aß¶×ç' ÍÌÝW[f&ÃIhñ3Ëlwèn $»›dlcÓÝI6t~ýêéÕ+w¯€0‡±»uG I*"æí°p´JMR’…S½ªéµ@D¬Jm’ p×YÝ½m[wûí·Uåî †Ìl¢!3™if-uH3“QRf’^ÝcÀÔif™ÓdáŒUÎ\ÞHµ…X×::ŸÆ» ÙYa‚>£™QX5»Û@3“$€*¹;Ä‚Æˆã8 º›ªÄÏÜ½»Iª
€»S0³ìº³Ãgtïn $q×-Éi"H‘¬»«Jwüß$0[—D2»
 HV
€$’ˆf0ÃCvu«³Ìèî™iøÌÌîÇZîÞÝ’Ì€™e.3›sšYD èn=ð3 ¢³è–¹ÆÖZfFdwá¡úplÂŒdw 9önafÇm­UÝ·ýéLrÎ‰j’’ú®ªQÃÍ9oEå\wÕÐ¶mkÕ1gW™YDdæÉ}ß6 e•»3L$Z´-\>ü—¿üêë¯¿šª×¯_¿{z-3èûqùî‡¾ÿþû~žG®µjµ‡h­ „
s˜HŸsî’¶áÕ]U
ÓgÌÌã84{Î©w’ÞÝû>ªJRfšIIh}[ÈÓiëŒªlÕ4pÛ6ƒ³k­ªÚ¶mõÊLz¬µºÀðXk™¡f§ÓÉÌŽ™ NÛî»ïÜ¶¨ZB¹í@W© £Œ[Uuw¸£3øwÿý¿	°Ny’“[Ø™†,k…î¶®pG•›I")É-èLÊÌª
@0
XÝ·ÔZ›¿±ð’]ðò43aí±›Ù±êÎ"™j³ 	©²AÝÌªJ€2„ŸÄ.5ëzkþý,ÝŒ?îöç\?\ç$pÞ"¶±¸ŸñæÝ»W¯ÞýÛüXa_þâm_Ž?ÿûwy»‚µnO;¨³…~ùðêÃñõÊ×Çûèl`¸1³ýðN RcŒSêq>‘­‘õ~W¢g)×É÷è–ôâøKáßn—§×¯†éKQz²¸eŽ}”Têz¾^ ‹D6df’Æ±Ë¼9Dolf§ÖFO³lýyætšž<" ­.I.il[®ÕY>BRXè. ÉîÞbtwD¬µ$ÁHÆT5dfnQ ¬pg 	€ÙÕÝ±oÌ¾Õªîmc·„î&*|&{ÈL¸¸:%˜ Á$e¦»-'¼û    IDATEÒ„î&<†Iên3 	À£»Uð.¶±Ö2³ªZkm1HØCfâdDHÐÝfæ·y˜Y¨¦ ãiÛ3sŒÑÐ¬5ÌVkp÷îŽ‘—ËÅÇˆI™‰–AcŒóù,•™1.—ËÇÏë˜(e×ù¼¿{÷å§O‹cÇz~~^kU÷œóõ«W^nW’cìûy{õæé_~qÜæíx,g®Û1ì*åzûöýù¼›s­ÕÝsN=lÛf« w¯Tw¬¹æL©îÜ)»sºûë×O¹´ÖºÝnîÞÝ·Û-ÖªµŽmŒ7_¼õúôîÝ»ï¿ûñ§Ÿ~&i$æœ!É€0w7ïÞ ø÷ûw§Á¶m­åî|Ð]õùÕÓ‘«g_¯G¡ ¬µºìÝwÒkeÖÜOO§7g¹PUŠ±GÉº[ùòòR%’€~õêÕy?Yðîùùòü|±;HOO'×œî. ªÀc¬µ Œ1VWfî§§ÝÂFüú7¿ýÕ7ßÌ9×ªë<ªJªÌ.6²”µæ$ÀÝ;õüáçËËKÐº[Æª’äî¤Ü="º[†›Ù1§$#sw‡EU­JIf6<$uwfW­cå¾ïÇq\.’f¦ê;> zuIr÷ˆÈÌîÞ¶-»ÌLRg‰ z˜«ÛÜ7ãÓÓ$f€¤ºÍ]’»õõ/ñ‹_\#".—+†³øÝwß=ú0Ì+“¤…¹EÝ¡o·›$Òº{?ŸŒ±o›GŒ1ŽãÈÌ9'€Ó¾ã¡ªº›¤™IÈaºëŽ±“J5 ‰¤ ˜K°1H<¿Ü2Óm?ŽC™s×mÛ~ùå/ºÛÂj•™‘¬*w_•——Û‡ç« ƒ£ó_¼uÞ»RŒq[³!´DPVµ.·kD\³"¢  Öj‰$ 3`T©ÚN[Uå*Hf6çZÈÃä¶m¯N»Œ›$šu·ŒÝ%‰nÃ#3Í i¤gN î.)s¹Uef’ö§§Óùìc3³Î23’ T]a£;éÖÝ0ªZRwfÁ‚"B)d¯Ê-ö9oqÚø )"j%I»€¤o¿ý–dDT•Ã›D²»éVU’Üdu«áA[­ª>#ifîžk™ÚÌ "i`v¡uGr˜gÍî>Ÿ†Pµ…;€$Tí# df7ÌŒ¬î&	€t Ý=+Ç]\kExUW¥9ÌLII|@µû YUÑÝ| @AÉ9§™¹; ©ªÄ;ñîX‡ÈˆÊÌD¨@wý@wÜ±Ý½RÝmfè`fÝíî%ˆUÙK
Ý¡E ÉÌ°ÿ4Á[³dY–ä1æ\ko?'NDveVuUß ÑÅ `¼Áÿ™!!~¬ºÐ¢+«:³2ãâî{¯9ÇÀÃ»ëûö½»«*3#B«#ÂÐZkŒˆ PU 2Ó†íˆ@°W	Ž m $»;ÀIÉ¯ ØNS°¤&3Ó#.sÚ(Ž/_¾ 8÷-Æ@Ò`t·ŸÔ™¹Öª¯ÎµV÷êîä0±¾j 3ÉÌ\÷Û7ß|óöî¥ÏÕÝ DDDw“)Õ~yýõo~ùí·ßJš—ËÛþ‚`=œ