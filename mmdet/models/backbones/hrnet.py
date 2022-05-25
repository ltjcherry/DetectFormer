# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import ResNet
from .resnext import Bottleneck


@BACKBONES.register_module()
class RegNet(ResNet):
    """RegNet backbone.

    More details can be found in `paper <https://arxiv.org/abs/2003.13678>`_ .

    Args:
        arch (dict): The parameter of RegNets.

            - w0 (int): initial width
            - wa (float): slope of width
            - wm (float): quantization parameter to quantize the width
            - depth (int): depth of the backbone
            - group_w (int): width of group
            - bot_mul (float): bottleneck ratio, i.e. expansion of bottleneck.
        strides (Sequence[int]): Strides of the first block of each stage.
        base_channels (int): Base channels after stem layer.
        in_channels (int): Number of input image channels. Default: 3.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import RegNet
        >>> import torch
        >>> self = RegNet(
                arch=dict(
                    w0=88,
                    wa=26.31,
                    wm=2.25,
                    group_w=48,
                    depth=25,
                    bot_mul=1.0))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 96, 8, 8)
        (1, 192, 4, 4)
        (1, 432, 2, 2)
        (1, 1008, 1, 1)
    """
    arch_settings = {
        'regnetx_400mf':
        dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        'regnetx_800mf':
        dict(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16, bot_mul=1.0),
        'regnetx_1.6gf':
        dict(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18, bot_mul=1.0),
        'regnetx_3.2gf':
        dict(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25, bot_mul=1.0),
        'regnetx_4.0gf':
        dict(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23, bot_mul=1.0),
        'regnetx_6.4gf':
        dict(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17, bot_mul=1.0),
        'regnetx_8.0gf':
        dict(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23, bot_mul=1.0),
        'regnetx_12gf':
        dict(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, bot_mul=1.0),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet, self).__init__(init_cfg)

        # Generate RegNet parameters first
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the' \
                ' arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise ValueError('Expect "arch" to be either a string '
                             f'or a dict, got {type(arch)}')

        widths, num_stages = self.generate_regnet(
            arch['w0'],
            arch['wa'],
            arch['wm'],
            arch['depth'],
        )
        # Convert to per stage format
        stage_widths, stage_blocks = self.get_stages_from_blocks(widths)
        # Generate group widths and bot muls
        group_widths = [arch['group_w'] for _ in range(num_stages)]
        self.bottleneck_ratio = [arch['bot_mul'] for _ in range(num_stages)]
        # Adjust the compatibility of stage_widths and group_widths
        stage_widths, group_widths = self.adjust_width_group(
            stage_widths, self.bottleneck_ratio, group_widths)

        # Group params by stage
        self.stage_widths = stage_widths
        self.group_widths = group_widths
        self.depth = sum(stage_blocks)
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block = Bottleneck
        expansion_bak = self.block.expansion
        self.block.expansion = 1
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels, stem_channels)

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                if self.zero_init_residual:
                    block_init_cfg = dict(
                        type='Constant', val=0, override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.inplanes = stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            group_width = self.group_widths[i]
            width = int(round(self.stage_widths[i] * self.bottleneck_ratio[i]))
            stage_groups = width // group_width

            dcn = self.dcn if self.stage_with_dcn[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None

            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=self.stage_widths[i],
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                groups=stage_groups,
                base_width=group_width,
                base_channels=self.stage_widths[i],
                init_cfg=block_init_cfg)
            self.inplanes = self.stage_widths[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = stage_widths[-1]
        self.block.expansion = expansion_bak

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def generate_regnet(self,
                        initial_width,
                        width_slope,
                        width_parameter,
                        depth,
                        divisor=8):
        """Generates per block width from RegNet parameters.

        Args:
            initial_width ([int]): Initial width of the backbone
            width_slope ([float]): Slope of the quantized linear function
            width_parameter ([int]): Parameter used to quantize the width.
            depth ([int]): Depth of the backbone.
            divisor (int, optional): The divisor of channels. Defaults to 8.

        Returns:
            list, int: return a list of widths of each stage and the number \
                of stages
        """
        assert width_slope >= 0
        assert initial_width > 0
        assert width_parameter > 1
        assert initial_width % divisor == 0
        widths_cont = np.arange(depth) * width_slope + initial_width
        ks = np.round(
            np.log(widths_cont / initial_width) / np.log(width_parameter))
        widths = initial_width * np.power(width_parameter, ks)
        widths = np.round(np.divide(widths, divisor)) * divisor
        num_stages = len(np.unique(widths))
        widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
        return widths, num_stages

    @staticmethod
    def quantize_float(number, divisor):
        """Converts a float to closest non-zero int divisible by divisor.

        Args:
            number (int): Original number to be quantized.
            divisor (int): Divisor used to quantize the number.

        Returns:
            int: quantized number that is divisible by devisor.
        """
        return int(round(number / divisor) * divisor)

    def adjust_width_group(self, widths, bottleneck_ratio, groups):
        """Adjusts the compatibility of widths and groups.

        Args:
            widths (list[int]): Width of each stage.
            bottleneck_ratio (float): Bottleneck ratio.
            groups (int): number of groups in each stage

        Returns:
            tuple(list): The adjusted widths and groups of each stage.
        """
        bottleneck_width = [
            int(w * b) for w, b in zip(widths, bottleneck_ratio)
        ]
        groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_width)]
        bottleneck_width = [
            self.quantize_float(w_bot, g)
            for w_bot, g in zip(bottleneck_width, groups)
        ]
        widths = [
            int(w_bot / b)
            for w_bot, b in zip(bottleneck_width, bottleneck_ratio)
        ]
        return widths, groups

    def get_stages_from_blocks(self, widths):
        """Gets widths/stage_blocks of network at each stage.

        Args:
            widths (list[int]): Width in each stage.

        Returns:
            tuple(list): width and depth of each stage
        """
        width_diff = [
            width != width_prev
            for width, width_prev in zip(widths + [0], [0] + widths)
        ]
        stage_widths = [
            width for width, diff in zip(widths, width_diff[:-1]) if diff
        ]
        stage_blocks = np.diff([
            depth for depth, diff in zip(range(len(width_diff)), width_diff)
            if diff
        ]).tolist()
        return stage_widths, stage_blocks

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ãí×wÄ{&³;ÖßØ››{·4<:8ô,S+dÇ]_·&.Š*[[[£oQëõÖŠÇÏƒ l¬¬Ú¢”¿®¹šØP8=™†”?¬õ ;	ÀÜdÑ­ûæ  ¥Œ§g‚›D_B Î/b•Wu²n\f  X«Û´[ßA ®Ù	v·ÚàîŞ–µ–›N Ë$MB«ó*Õš'óÈaW÷„€\XZòûE°6¿`KË+¶º¶ê$ ûªõ@
.ë^«›kDyq¥iåEw'f[uvÆÉ>Ö#r‰+*I SŸ«zğZ2„UgfÌÿU‡²îPÑù'C‹E¡PrŠÓ¾l´y­OEL©eí3aÙ×§“ıe›œ,ÙîdZÇ’rRÚ¤ê0¡v™,ëz3%+Ÿ¥P°’¶U«3¶¢w
©º±²âX]^¶%)ûóµš“S­mg‰DX¢HÜšH tw 7! –r$`t¹…¨k€BÌî	@²ŠeÆ1JDââ€)¼gÿ‚¤”®¤¹ëë~}È9bü‚0‚4Œ š4©#`$õu`ıæ„ŸÊ„ ôìÅl‹ËŸ€™ÁÚ‘X‚Ô¹ª¾¾¾²¡oëÌ®Î$ùŞî_kL¹9¿tëbÜMÊÓ%Mèc–ŸÌ9¹ù7W1 !ç¦İ
p}«aK+•¶ nÀmpmÎ:f~±lÍ²Í7JV™ÍY}¡¤şZ·Í¦`-»gÇš@OOOíôêÚôí/,.Ù°Ú2€(ñà×`ˆSˆÀHûgm"W²É’¾çÙ%Ë×V­0¿f™Æc¼¾o™…oXvéØ‘ÖrFåxóÈËÔÂ¾¥ê{6¾x¨m»–jlòojÑÒÙ¢Œdõ¾‚µ_$ !ş"Üê¯›lÄ€ºrWàÔØ„êW´	}7Y		Œ+é‰œ¨´õ«¿@ DAÒQFDÂÒÒãò!dé8	½G1µpG0^7€wmpÜñœöyºÎo€/»pî·n=È©?UmMc[£†;¯ÆÄùšÆTM}M~fÂ¯^¯j¬­é¸ È¿ù9õ7' %„iL„ <Ö<q¥>tqtd[/JÙ		õ[Ò˜Ö'
¶£qi²l{¹i¡l»ÓN n¥§lo¬`»©€Ñ¼mO:	¸<µV_ÆÖF§„¼­m#S¶ÕTÑ–GòÖšp@ R®OØêpÎÖÀÈ¤mêziõwâ	fJN îå*¶;Uuò¯Y¨ÚÌ¤Æ\íOæltP‚´®5 {C Ïòï7À¾ñªõ¥ù×“.„ü‹àĞo€EG·= ëè·^}w/ì¹ ¬Ôºˆ÷Ö#åÙ!åHi%vX›Kˆ8·VSŸjÃûn@$§é'ŒÉãX¯i¬p×q)ÜïâŒÅ!1]áS¿‰ ‡]€htòïk2ÕòohĞ	@,cÒÜ„q% $ îˆÃRºÓR®Ç°ÒeÜÕş>=[?ŸU7	üd¼íÅu(!Ä ÙÂsKñL3ğ˜üâQAs%RJädS°êMÈ&')P‘<
])$ÄƒòxL Cù¢äšÒE	Ÿ*ãqñØHºAĞDğs’ÍºDiŠd#VAøJŠ@ÙÑ2%®w‚{:)éÏMu¯^g¶Çã¹·êÇsD°í™êàĞ=Üâ¬M òÎÉ4µ#w×“ğç²[ë|~/}J%Ây8>\3AT)çûU²-®Sgêß¶ºá-eü!
êìG©à~Q Çµñ?" ¹ŞŠ Ô²¿Ç÷VP?&v]ß3µ«”®?êyÈ?Wt?_çòİEK—hñG[·¿i!Æ<ÂíÒ¿Mt·¬7I ğvX\½¾½í?ËG%A2‡ñHOĞİ§~¦„}:.¹/õüLJ”©v–EµW'¢»ÕS=+ˆ–¯X%g2V0® ´A bµñçÊ¦Ş•#i'J¾7·tÒ§ÇI@,¼ÜÊ+!ş¢‰É'„òïÿ‹Íÿ—§öÏ¿°ÿËóªı¿¬Ø¿õìØÿ£rjÿÏÙsûÓï»ì‡ß?µCÍñ€…Ôˆ“€mË`]3ÜC÷ÕØH¶øÜä¤€ÍÖ¢{~àê{| ìĞ.5×1ß]½zpr¬m‡vz¸o'{ÒA¶mgÓ¶I©¹•¸à„#È"AcÖîÇÎœtÜŠÏ./ìæµt±;ÉƒÒ-ŞH¯|xëF‡èb‡—v|t%9íÂÏİcœ&À‹o®³³vê\8ç.o_ÛÉñ©dË}[V_ÁÍ¹Ù‰<É!Á=VãÑSú¾”kÆ|È>ÍülwPó ÛcéViŞfnüĞ¹[°¾#{¹“s»Ø— ×19Ú«ùvPrÂ€cZ(¤{-;Úgc/ııcêîşê¸Æp1ş©Ÿ/ªo=µ]3Ÿ·lZóK¯é˜~ƒ50ÖÇ¡ßª¿zÿd %ßy³Ã8Ê·|X j½ËÌ0¦‡ñs|¿®×ş^t’dˆû¸w¸ƒà	™¯Æ» ë'²â7H©ãß˜ëyoè<Ä³&2s^ç÷¾Qêëˆ‘Nò/ I]Øß~†ÇíN 
¼@·öÓ¼	@–»^ª:†ã#ù:	ÀHşEÄu’c‘ü#›Ó˜2i³³åà©²øH îïïÚñá¡cg}Ó	À²ä{@¬Q! }Ìç¾<›Ê@†%Ïäí¡ºÑÎ@Ëa.¦¾a¤X$ÿœ`¢ä!\ó·A›$ıBíïM{?¶ù#:	upÄılOàı@h+Ä÷Şur.õIã<Ÿ/Uv">[ûãıÚçÄ6û|{¸¾'IÛáËë<^_¥æ vø‚ ôşå ¤?ãŒú;}^}ù#üPRİt>sxnÕÁ—C]b¿íñØ&”Ş	~wy²c7g{vwµa·kv³aïîwìöjÕNëv¼¿¨	BÊAõ[RØHg¾¡õÛÙz¥ÎyÓÆÉÑ­½ºµ‹“;Mwš8^ÛùÉkM.d×Ô ~zkWWnµwv¸ãÁÕ·6ìÜ]€OíúâÊ®oïŒ?CÇW—öêš€—v~}mç¯oìêşÎî„Û;\Û­¶İİ]ÛÃÛ;ûğáÁ¾níööB“Å©&#M*nêN°¹µéÁ€WÖ7¬E
òÕU¤–nıG–±fË± …òmÕÆ¢ÇğË!©p‘­9p¡…œ ÎDş‘ $‘G'ˆe¤àÜÂ¼•gg¤¨=ãg±8c$å€ œ­ÎY©TrË¥-Õñ@õn€z–İ[j5­PÄÒOÇaå"…•¸}“SdâË»1q6KMİ‹}³^„ªU=;ÿvLôw÷5±¿JphÂşáíìzöVâB’fcmÕvÕ~9$$ÙEØZóÔ÷(Üd²Â’ ÈeÕkzº¤ú©-f´\){]Kè¦«X7×âùÆ‚×¯ÖH ç¨ÕI€‚ò>'èÙæ°ØY"?ï¤h‹=µ7Ë~®ÎYP{´H¯¯º’nmcUïšŒÕë¶J èeb´<ÆË¾'§ CÜ=ßÆŠöKÙÇÒ‘Lh€¸‹µI@	m0–%.<ôãŠËŸD,U‚e€Lƒ ô8SÂ˜P qæ®Â©€t
Ë¿œµñ1­PÇk?¬ş¤œ:é‡’úHø9açD –“m`ë>)¼Ñ]7â€ì‹V~m@üé9"†£L«ş)Õ›>_Ô÷P®Y©P±éâ¬Õ*uµ©úî¬Õgö<ƒxMïŒ˜'X’.ê=CşUˆ§V¯ØbSï´V´ÅVÍ6¶Õ§Z3ú¶ŠúË¶¸<kÕzÑ	¿åUõ‡E}/sdÍªX½Q¶™ò¤Í×J¶¶¼`[ë-O>‚Ûü‰î}¡±åäìÂv$(Îè»äıtKÈ#¶\Œu©,ÿ"Ñ†•®Áéô¤¾Éš¾ó¦¾[’ê´¬0³hÙ™†eÊBÃ²•EËi_¾¶l9íÏVš*—U.iÀ¸ÎÉh{fvÁÒ3óV(-Xnª¦1 `ê¸÷¾t"2€ÔÓ³œÆõH &–†éñIÕkÁ–V­I,Rbò£±dsúŞ‰ûI|.b¨Eâî‘Àë$^dJ¯”,¬h~M ¾ˆ.b€€sÚèõk9¸n‚Ï‰¼ÿ$â9_×Iùk<GúHŞ“+ÙÜŒÆ¾Õp@²şşš ôx€ À@şXÊ
ÑÙş»;•¥<½Q*Ûü”ú£¾ª¾“Yõ™ªŞ	%¨u	*_X­kÀêZ¯w[½gÔæUÎvi_ÏˆU‰ëç±ı†mFÇUtÉBfµØf9‡ãº†¬ò\×ŒËºW¹»ß*Z®vX¹7e•>	•YÍ3Åºú—ú`¾nC… ,øúpñ-.8z‹G_‚şB‚©G_aÁz&ç¬W×`¹O×ì-Ì{ÌÀ¾Ö€U]oÖsëÏ–­orÚzõzsEëQŸîËNŒ¥¬?­ñÂ-¥ïI
=qµrmJB¥Æ¾¼ÆŞ‚Î!ÔD!›S9án]XõÀroŒ1ÕA‚aGZË\‡X[œŸòc“îò—Ÿ şÓˆ”­^WÈQèˆi‡Ö”8·ÀĞ7:- !#HÒä		¤Œeró4~e''4>âŞ:h©LÊ&q«è[d®ó^.âàáKBŒHtD‹5Æ^)Î}XAêúºG†,àŒ±Ãn!‚+ä3)jÑå1f'¦Ù_¥Äñ-K€$™J“'U@Ôvîç.»:Fêà.™¤	™Ô·^Ü1£•¥Çseu”Eİ³Ãe´_Âíˆ®=©:¤Ìf$ÌKğì‘ÀÙ-A3!níÑˆ[Wè¡|B†Ú…Yˆ,İ+’»€u'?µ/¸?p‹a¬ã“äºAX¦Ôñ\Wû"qê{ÚŞI a]ãVv\?ÙÑ×ÙÉ7¯o¯?Ïïç`Å@çúı%ÜS?ñãƒîõ:ÏĞrÜÜƒå ômR>¾$ £òÀr'YÌñX<hLÏjÌÓ³Oü×'–ı/Omê¿>µâŸÛøW/,ó¤Ë†´?«o‘äsšÓ›XóKÖÄm‹UJb}»uÙık{ûæµ½wo¾~°Oß|mß~øøá}Àû¯íÃ»·HìíÃ»Ğ|íNğJó7ñùHøá½ıæ£ğ!àã{ûæ›÷ööë{»—ìÿ€€‘Àä×='» *½ıi³ş"ÔömÅ’eúÊW*”®'„/¡Ätw-¾3”5µ ¡K$×cB˜˜èÅ]6õR	Bem	ºNBÂ¸5VBœÇ¤4C_}ei½›­?vÛEÖşú$cÿü4k’¶¿=IÙÁWİvğÇ—¶¤±iMï¢œŸò„È’É•óÆÆ%§
ù3í8%½ŸEÉŞÈÄë’wHX$YÙ±¹f»’£ÁÆJKó`Óµæ¬Q#³ú´M&mjB2l³ø¢!¶ç€¾‡ïs’Ë[HîÛ!áŠôdÁCá`÷H2á¡ mïÄN¯„“Ã3;UyJy|&òÂ.O¯íêL:`sşo°éœ‡'’Ù«69N†âIw³&#^:ız{õ~bè€hÉ]Lù‰Ãxç^>ıX“#oéœä=õéİ0¨Oï§_ãd_ø´œêëş¹Ñ+e‡­’µêTÊÊ}6•ê²ô@—ö>µº¬ÌÕß<¹îÑ„»x$‘! KI†£¶Å1X$j<ôo<|Ë‘Ğ¦¯Rò=»E«±úÆuíhøL×…„âçıŸQñGL‡Ú„FB(¸äM"ÁÔCø’ ü£öÿAÇSF"ğK"Ç p_¿w Û8–±šxÏè<}z¯Ïkjÿ¿  ¿"kjR?–×ï³k?Ş÷±şá˜D z}ÁÉ¿zOÏ}H º›'ÄŸÚ—¶]êCüÄÀú¹ _˜¬PğXÕú¾›6¸'›pl`{m#€…²úø¸ú6™€q‰f~ÃÚ’vÕsÅzğ|Ş&Iİ’÷ş²‹z	^wÕSû:‰±j7~p=şÈø*!¡ ı* `IĞ1DàçÛCİT'¯_hkïOÛu"ã‰U¨¯ÆÔ/ÁüöxæXÀrò|hß'ÙÏ‹íñÛ`¼¶–¿@œKKî‘Ô1ü £KÒînÉE=Ï4şkœWŸ–¿İ¦Ä½ôm{la]óqLHê*ø7äË÷şlw°L½"¨_¸øİéá¦İ^ìÛÃí¶İßlÙ×÷;öğzËn.W4ğ6íôÒ¤&¥hŞªÜ=+®ìÏL7´Ş’¢¿l•iMüR,
ùšÊªÍ”ê6_nZ}F
èlËÊRÊÅªÕf¶²¢ÉfgËNö7 <Òdƒ àåé…iÂ?&Ãîé‰í¼²ã³3»¸¹±‹»[' _ß‘YøÖ®o®œk²¿°››3M
çNíìôĞÈV¼§{mÛêêŠ-Ky[Ñd´¢e°¼¶î	7æ——­¾²â$`ksÓ]}ç–È¨Ûİôe€0 VtM²ù©²KÁµ dãŒ»*€n}U)‹¸ CÂá†‹Õ^¹¢íN j{X™U;‹	¸îà¾Ú¬“ äï]A
Ä¹Y”°¤Ëæ&5—u­E[l-9É6£kBJ6šM[#Û™„£İı];8>²“3’s;=¿°ÃWGvtrìà =µßÉÑ¡]è}\K »½º´ÛkÜ®4ŸØÉ‰Ú²PÇOrc Ë«¶¶¶êí¾ù´N6¶e¡¥õ–úhba)Á±¡zêKë†ˆWà# ü°ŒD¸(Wf5pÎXEÏ>Sµ
$¨³.A¥±´è’€[Nj0…@ÕóÛòL
ı)n}*IRAÜ°})úd‡[ƒüÕ½¦õ.‰?ˆ’9<0ä¤šc ”ÑÕ6_‚Éjd8XòE·^âY¸U¤Ü`ÀP¿¶k[Ì^]‡±öÃ’0ºöúuu¯°Ì½#‰êİ€c=!öp]Àê%dËr"0Z.Æxè$U§M8‘ø#2#‘9–âï\F‚—&éÚiÔÈ´[­Ôlu…ÄBÛ¶©~WŸŸ³\6k¥Â”aA:?‡uè”ÍÍÓ_g­<;¥÷‡0®õÓú6¿8­¾Qé  çİR°RÕ5–fÕOJ6]š°¹ZÉZ‹U[m…,u8ëË«¶³¶¥¾°å„<îã}}ƒNt9:ÀˆH BşEpl,g}Õùe}»+6»°f3ó+VĞrAeqaÕ¦×}åB}ÕŠõ5ËÏ­8	˜«®ØÔÜšMi{^ÇO-hÛa VÔ›–Êä­o8-á`HBB$ûT/¬˜ +‹Äv={°T¢¯¥,;QPÛĞ^Ç ÿ„…%m‹-O84¥±`(•²—l#Q÷9(t0’yÏ49(!# (ü®ôC
&çÇvl“Ş—$Ş¿¿÷ol¾ïKâ/‰7øLË Wm’Rÿ+Lo§ì?$ ùy€"ÒX¨Ù’€ÁØ­ÿ„HBrû‰ÿ·Œr¤±èp{Û­ÿN5Fğ¼ªqÙ	À¢œ¼·á1+é½”÷Úô‹^/Kª[ùY¯Íh¹¢²ü¤Û*O{³Ïûmæ©¶=ë³òó€m+©,éØé.-¿è³¢–ã6/”unñi·•t­Ò“+<ë¶©§/µMËºÏ¤®;İ3juõ©£Sïw¶»wm­ı+[>¸¶¥½K«oX~iÇrM¯o82së–™ß°ìœ–kZÖúxmÍ²Ú—­oZz~ÕÒs«6 «¾Ñï$÷LÓ2Õ†kOÏÖ}9§yi²6o“3š/İSï@sñ?¡ø)G ~Ÿ1ö»lGaHD@şµÉ44‡4[¶º´ìXaİÑ²å¥%Ë|ókË:®¹d-õ}ÎMÍ…Äû"	Hø±ƒâÚï ÀË…0”?!Z F·^Ü‘3Rzc‚ê?£Ê’rN fóRB%ÌC 6š¶!ezckUÏ\qa¡D2”ŒÃÄ.ñŒº §kgqWtr.Ä9Ó· sûG‡m"¯9^mJ†â.HA)z(¯¸6óÃˆŸHüâñ^€X†$FÂİ2kƒ~mNØ¨ö‘}Ş-èâÏ	„l/%pCi<€p‹n‰O¸Z*8J«:¾WB& )‚"ÉñXpBÖEr+gÀïó¹×I F’mÇ±Ì¹ ˆux¼ÇüÔNĞ·'@èBvÀC½béÛ%ÇëGâım%‡óUÇH Æk>
õºû’u‰×qòıÚş?C bíHì¸	¹€Œ3ÓÏzü'Æœ“¥%GÔ$#ÍK&k-.º\¼Gl_¬WYÿñÇìŸşü‹ıòËŸì¯ÿô‹ıí¯¶ıÛ_ÿç€ùë_´ıÿò·VùWëFr‡UÉ¿[[;vs}cşE×øç¿éüşıÛ_íŸuŸùÉ¾ûá“}÷ı'ûøéƒ½ûúÿÌÎ
Şïy>Ú§³M?k[)Fö»H9¶%¤ŠjGHwıU?ãg‚ƒoÌ·CäÄu~4?	"ù¬°tmspñÃ_=±½“æï¿²µÿå¹½ÿısûö÷=öÊO¿j;ÿ»§vøU·5FG ¬I× >÷ÌtÅå®u­k›¶¼²®¶˜v™¯“’æ1<0ˆsKÂ«µå%ÉíË*%7ƒfÃ±¨1n¡H¿Z¹d³xÒLK¿)LZ!GÌm~æ`5¦q”xÒ!