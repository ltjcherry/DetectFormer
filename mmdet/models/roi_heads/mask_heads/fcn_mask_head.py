# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock


@HEADS.register_module()
class GlobalContextHead(BaseModule):
    """Global context head used in `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_convs (int, optional): number of convolutional layer in GlbCtxHead.
            Default: 4.
        in_channels (int, optional): number of input channels. Default: 256.
        conv_out_channels (int, optional): number of output channels before
            classification layer. Default: 256.
        num_classes (int, optional): number of classes. Default: 80.
        loss_weight (float, optional): global context loss weight. Default: 1.
        conv_cfg (dict, optional): config to init conv layer. Default: None.
        norm_cfg (dict, optional): config to init norm layer. Default: None.
        conv_to_res (bool, optional): if True, 2 convs will be grouped into
            1 `SimplifiedBasicBlock` using a skip connection. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=80,
                 loss_weight=1.0,
                 conv_cfg=None,
                 norm_cfg=None,
                 conv_to_res=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='fc'))):
        super(GlobalContextHead, self).__init__(init_cfg)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.conv_to_res = conv_to_res
        self.fp16_enabled = False

        if self.conv_to_res:
            num_res_blocks = num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
        else:
            self.convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else conv_out_channels
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_out_channels, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()

    @auto_fp16()
    def forward(self, feats):
        """Forward function."""
        x = feats[-1]
        for i in range(self.num_convs):
            x = self.convs[i](x)
        x = self.pool(x)

        # multi-class prediction
        mc_pred = x.reshape(x.size(0), -1)
        mc_pred = self.fc(mc_pred)

        return mc_pred, x

    @force_fp32(apply_to=('pred', ))
    def loss(self, pred, labels):
        """Loss function."""
        labels = [lbl.unique() for lbl in labels]
        targets = pred.new_zeros(pred.size())
        for i, label in enumerate(labels):
            targets[i, label] = 1.0
        loss = self.loss_weight * self.criterion(pred, targets)
        return loss
                                                                                                                                                                                                                                                                                                                                  L?3??4+*,???J=x?N?k?s3S???q??U???lV]?d???N?K??????C"?>b]??! v??BO?O?6?G?W???O??????on???G;?????V[???^?l-M+???6f???Y?????74.?s?w????`?`????]?W??:-J??2?+?e?:????5[??`x???H?L??K?7????n*T??1??O 0????q???<n..?/?????#9[?z?YU?2??????$12???E?~O?v?!????d??X?D*??'&v?  ?3??4i????Y;4v?kYj?2???,po?????Iv-^??9??6?b+??0a/+?x?; ; {+?????`???????a"?  ??#??F?Wc??.K???yy??C+y???X?Y??0? ????o?U????,??z	???K?#???o?^????~}???A????5S??Y??!????Z???Y?u???k??-??o?3 vK0??:6?Y???????m_?;I??R????????K??t??????~??L?%??K???N???Y',?g/?????!@??G@?????N?.?
S?????u??~???n?????}?????7?^We??W???\yeo????[?????"bpN?~}	???o??/ ???@B?GV???????G??1??|????fw?r ?AS?????{??=U???Y???Z?:?g????g?mv??,?Y??*oQ??k?2???[???(????]??I??V?u/??X0???/????E???.???0-P?~	?j??  ??:?????G?_|N FL?}?8???w?????8?!//??? ???? ???9 sz?ep???_?N?;5=<=3?% ?,??????????L? ?????.;OMz?{??_N?T*?B_Ao?D l;?j??h,?`????d??	??p??w?????mW0X?????q? ??P??^???=???j??L?	3 ??o??D*f`e??E?WZ?2?c???hL1?d?$??# ???:??C?????????5????P
?????+???????$@?C?????rT?T\??z;????nz?QS??z1z?xo?26??:??M?:E5?g	?Q$={??^?????????U?b]20?1?_???????-{?????w;?????`,??,???>??~??L???l?~?}??g????$?X???_)???ue???e???``??xW?M)?4??8???U?/?P\1?\?'?zP?B_???????]?C ? m9?}???? ??i??????????t?R?fk???N??{y?????p q?+??T.4?q1?z?^v????m?7???;???G\J????JJ&3c??i?^?u??4+(N??s??XKO??4??W????????k??h??Q?;?z?o69?j???_???b??????N?)h??`? ??1???.?c?|'o????L?%z?[u? ?;??yo???o??"?T???L?: ??.??o(o??d?/Br]?-?U?lY???\Q}? ?i?z??@??h?o??L???????cK??????????K????????????_~???+?nFr?bG_?b??
:??^??X|??1b?.m?xY"sx?????:?:?J?S?=?????7??#?~?????f???w?R[W`?;??\?K?7?VWq????3??t??o?????????,/h*??L?????@??\\??dw0??~1/D?????f?yyP??e%??T?Cw???~y???9 ????G??,????nU?T??`;q*?JiEZ)vu}U?Yo?[?gQ???????R+w]R;b??W??Z)suN??n??[,I??X?7?z{?5?)?9B/x}???<L???????E??  ??hb?87D??'?|?????"&G?<=?ec?	@we?)? S 7???,???&????  om?`??`?' ^?????????????bnn ???yw?#
}?k?5cF?&[?oD???5Hj????6;t?$????WWT?n?u????jokM??]G }'G{?\??p??????YaI>??F?omt7??vZ[????????m4fR4?!6F?^?????X?m?be?1?L]????1%5?t?U\?%????*???!V?????[???30?s???a???.??????Z??m?Z?
[??@??V?z??l?L?u?fYX?k?g2?.??B??G?/;??P???<?????x"????h????W?o????????5Q|?%??n?O???x?????????nwug??V????R?~S4?"???3????*???????X???|??W(?~r??%??^p@??y?aUP??L?.&F?
?XsQ?4??~???9????A~u??j????m??????"^??? ??<?X?q0??????/??CXiS??m2???h?(?r/????????6?H5???\I
???y?p?+???????_???3%b?X?P6??????E_????	??rN?8i??V??d????????J?v7fo?\w?b?L<\??4MS?C-?\o?L??P?}?svsx??K?7?R#???VL??????~??`p?*4????\?a3?????K?/x?z?;?W? *???+?fD#d???<?W???{5????Wt?H??????Zg?x?T   IDAT ?`?k?	?I???_????????????x???o???y????e?? ????`10qZ?N??????Y??m?(????n\????????x???_@?2B?_g`DL????6? c???\K7C??u?c?2r/?????Q???S?????X??`v??	?n?U???Z]???/?A?%g4Uot??????7F?p?l?* g???k?eUp2m?')???~???????"???S ?S?8????N???]??:?????)?? X?)M+?@?V??*??N!#??O????p-????g?{{??F??034??s?????????O?<?B???)????1??????S[??n???N? L??J?1=3B F ?33?????i???}?x??-?6?U??(j,w+
???=?"?_Y?f???K? ???H?`?No???? ??L?O?? ?,?/?w?;?k@???????????J??]m2pA?e(G:P?z?^7	?????8+T?&?e?f??.?D?q??&?0?C>????????? ??Cb?????<?9?[? ?K@???_?k????-??????!?f?????E?????"%;{?????3??;??~?E?????K .S?G*wI%?~??h-?z?????? o?5?????4?u???`J????'??yy!r??????Zi??DJ??I???ueA#??gqL?E=??y?!yp?R???n&'f??E?s?????????????0????D_?0????????H#???????a?%eCa?y?\-L b?W??~?????H??Q??u^?y^??0^$???????^qM??s?_(Wg>?F!???s????B?L?k
2???X={Y?Kr???r?g ?%??q??H?.z??Q?V[{(?%_??^_?z{????c=??G z????="?????)?xO??,u??W????a??K|??	SV?#'????u????$???Lo??r ???]??v?l?J?MQM?.??\???CXr??/Y"tvW??\yE????????z?6^?|.S?	?l???o?????P=???zw_??K?U????Us?????r)?V???H????}?????>?E?~?}#?5?L??fE2?R5o?B??BX?/? ????``?U?R???e?%??????????????ct?*G?z??:	{m?????f"0?????a?%?In?:
???1??>St3W?''g?????Wt(?*\<?K?`7??[???/OG??sp|z{???????:?V?7??z?lg???.??"x? ??g'??|??=U?'E????B?U ??3???y?9S??D?P????`?o?`?j?B ?-@Z?W#qZ?H??u?	?R??/???W&X?????????????g??????~???C???yb????G?SOf?Y???????????0-???nO???! ?????~[L0????[?vcq*???0?_20 ?:p?1#?T?$????gNU??K???p.v????U??
SRv.???;?J?@+?/ +?L?%???J??????g'???]???no??y??=?nn?W:?`?? ????I1?:?f{?>L?5& Sv#??c?,Xe8k}T ?XD????? ?! ?G??????Tb??c#C?_??N?Y?????????:g?o?3du?)G_??????-?P3?????????b??1	???~P?????[??I?O???oH=AZ??3p??$<F??u??cJ4?H???-zb??<}s}?????`?tg??[+???9F?+???r??? ?>????9?? ?6?h-??????????*^^\h.5?+?Kc@???W?2?u?{??~=??We?t??[.?fQt'?S??Q?JZ&?j  ?;	??t5xai??<?^[???)S???t???/?????N??!????M?`??r_V?????av=Z:????4d???????x1X???0?]o??????????}?p??[(;??s??J4????? ?;??s(k???0s?L??b?f?g???????_???|u??%?nh???&X?4???*???W???bi
?bQ???U??[t?U??`+E??"[XJ)??!???i?y?3?mR?3?W?k3???`????\4yR????=?Y?vT?/?P??^?y??Cp??T)Wj%??4K????3!.??????!?S??e???W??"N??-?r??J,)??NA?u?g+??W?^?$??f?j?d?wj?E?%	?w?)??????????2w???????l??0?W?c??w??~scg%?3????2,
??/????????u ???E?g?	???X?P????<??)B([?y??6????t?;-?q|yy????????<????????T?????5?|b` 0_[%??a??/9???_6?+L??/?R??tZ'?? ????d~bck?t????]?gJ??u???{?zmro??]?8f????0?Y\????? ?o????????????????N?~ ???? ?E`??}????y
?	????l????5d`???l?u????LN}1M??DhW?=Q?k?%????T??Db%9??s??; {?Ny?x"4S??K?XCG??M?B????????
}Cj??n????}q??i?|??[o.?~	??E?+@K?p<???K???k????g ???????:?j?z
+?O?%???|?W??n?*?????f?o??:??Lht__%6?W+???pV?+???(?R?c?7??{???|=????X??b>???1p?AUu`g6&???~??I?H???0.??????'?zqx?V_??g??x8?r?A?R???G{oo0?=d02p????L???xGfF(+???f`F_?????2]???? /(?=?]?{??g?2?s8???.??[=?(??????^'??Y?o?Z? xy?a1?W^X?u?% c???????\cr?????[?2???~?}?)W?$Nt????LW.??6T}??/?~u??????^T}?????VM?"8;NaN4%????Es?2 .{%?43??b?l??{b??^??T?y??????~?9 ????1'?o?E[??????{??? ?[5???????p%???? ?g??]z=Z? 5Z=??????0?
+#??Kz????~?*?e`??t ??u?{????4? c??t?J)???u?	q?e???d?? f ?_???W?\?8v??6???>?P???7????[??????c????q???~b?)&???$^j?.????[20??Vi??R??D:ESV????=9?????????u??????+'&f????.????IVs*,??6?VK??$?6W??????uw?o????l?{_?j?.????? ?m{?????<????OE??+??s?8??W1??cj????{u}L?]?O?q?O$???????	?????$kI?????:"?%g?tn???nJ?U??`R?]c?]4?^XU^?j{??7~???UQ?%?x?<??E?R??V??aY@??Y?\????I??ZL???6?k?????{!?9?C???????????6?s?#O????????????#?=?????rn??????x600?#I??m???????%?t??????lwj?2??}6>6059?MS???s?????~??-p>??`S-KGS?i?]?E???8\?FP????)????2?^B/k????????o?1@?x???a??????????????????fo?????^?[k50?G}?????6c???_n?aOh';? ?????[?zg?c#C?*?>?9??5)??,??@????u?V????WUzH?;??P?{? GC???6???????????\h??Y?v?????W??xx.*?TY?, .???"a?$)%Y?)M?$J????????L?I?u?i?????d?$O?????????`??7?o??Z??????~???v   IDAT{`?^g{u????\?\]\k??????g??%?+
?j??\K??J??m?.??g??????j?? 'K?? ???? ????{?o????f??*<????4E?j?s?W??!V??=?;??R?s#?@???f?1?A????^B??=??l??	?Z3??N?z?#??"????wa?????8xeiY??\?p??YK????e?_?wV???xh????g)Y?_?g)????p??????m?????G?Xz	1????u??B?OZ7???j????????wK*?????z
??9?us,?????f\'?i??4?????E	???M?e??l??ei2????(?????z?Ll??D\U&???3ZR+????q&?:?; +??P?]??\??????uT`?"???hk????/?W????/?cU?0?f?0??=?Z"0?^.l??????????~9<?b???n9??y/??Y????????6????C???e?^<\t~?)??Y]?5???]tV??J??????PgW\b?3 ?]?~????_???? ???i?6f]???z????<????};?????????uN?????H6????X?R???)?????]/;??px?W ??A'??T??????z????,??Ec???ZM?Y*?"&?2?
yid??uc?R
.?F	z?pKY?:+???]NM?h??M? ?{uuzyy?%k}??H??}?x?=???u[?s?O??<????_0??G?{??c?C"tc?????? ???? p???\???{m 0mE6?Z?I@/?11>?1??K0 ???Y?Z??c?H?Yp=#M?Z????3 &?^[4???T?oz%?T L?E`????dc
?
?a?/?W}??? 'e?{?w??s??w????]???c??L"0u?????f?_?m?2??9?Ea?U?zp?37????w?'??.?B+?o?.G??/ ??????&??T?z???????K??PDC`?_i?x?
?1M`6?=*[f&???^XQ([????XPw?&]??? ?I?????????)?|??Z???M&??????&?]c?x?x? o??~????????8Tv? ?????F ??i?? ?????gx???pki6???d`?so??????DV???O p?S6 V3$??? \???2?? ??_W?Y??Zq=?kd?}??l???????0C??5?a0ean??h??%O???T??@????5nX???????[2??$?????Pq??% ?G@?????(???+???)\?W?	?D??
;KoV?E??/g?i3??.???\J Ez?a?j?t?*X4???? |??}??Y?"???S??=???Y?RV?+p/7 ?b?0? ??	?YO`N??I[?: ?~?????o??"pSE?Xz!?(?7?}?<2?5s?u???# S??7???$???/???V0O?	?F??X?d???QA?u f?t?}??\?? ????H?l?_?e???????+?z?????9???-???I)????g????????HJ,?~??*	?????s??????W???cyW????????}??[~|?/?????Z?????x??o?J?5??~'l??????L???	?o???/o?rs???T?nJu?c?,J??|RB??V%B{??? ?i??.?????1%???t-?i???{I?f??z>????aRW|???h?)$?,????,?5??{????D?>~u????????)??C ?W??[????p:=[???=F?:??A????S/?>.x?????2?????|??-?8??2~r???l7 ????????g?? ?????x?q`??????gV????;7?0;??{>?D???FT?[ 0???l?7???????????!?/#??qZ?????4??????N?%???}@)????2??????k????e;?M"q???^  `w??D??Wx??#???NHr?b3$??2?xv ???M:`mo?????V????????V?A[?????x?jsq????w????u???+K+?????p?j??;?M??rldptt166<>>299???*?k]???????ej??K?uM?Z?Z:H;???K?_???'c%?w~n?[?(e:?h)????yj"6??dv???tH???Y????IXD??=X?Z	~??`\`b???Z?_:??P???o8?	?????????"????????????Nw???;[-0????c????z{???????\b??~%?'????'?K????r???,??????ya.`A????8??+?u .?omnn??????????tZ??e
???U????Dh???X?B?L????l?????????.:Bz??z??W9{??;EbsamBD???#TV_|???#<?/?.?? ?9?????e*?aOu?*?5n??j?*???Lf????????*??2I? ???N9???E?QJ????O?P?????JN?8???J??n?*?pj?t?5??^U _??b?_=???zq{??	?.??v?o?3??&?5??/???????o???B?	 vj??????P?LN??B?N?????w)??)?i??????gm?J???B3n???|??????_?E L?l??2`??_?Y??????By???????:?????qM?+?`?X?Z??;O]??"?}????%?J@?f??4?*?o??~?????????"??? ?{	??#??k?;?????W?????.????R?@Lo???????l?$???????~?K??iy??/8?`?p????X?o+~?t?+S?c3t?o:??v?:-? ??X0???~,E.?<wY8v?}V?
??u?c????<&????Gz?O?	??J/ ?9??l?????(?p?d?3_?K??7?? g??Z????-7L?T?n???????7???[CG????? 0bmuijrp??_??= ??/?7????????D??|??|??y?X?}6?? ??0?]\?
???`?????, ???T8G?$?8??`i?+??H8?s>??[X@oH???j???u??]??"0????H!??.?G?K?^\Y'$|F0's?D???????? ]??? 7?p 0??YK#???j???_????`oo#?#?p??????/C????S??u????!??8??D_?? ?N6?1c?I?Y??? ?}?$????????? 357
Y?`?z?^^	_?kL'?d&W??e2b? `?<???3? ??g?'!??4??Za???????"V???Ra;3????]?e???????????j??????? `??? ???????+??)O????oH??Q??T???}??'?????? ??????YG???7????W??8??l?V?*?~???j???E?]@/????(??????????????Nv)7j????`???????? ???$??v?/+?Y???>V|v2\???????u?w,/?????z;????b?????<??l*:??????qJ<T???x]}1?\E????=?p????????}??h??2B?2?'E???&?%?K???+?????e??#????O=]?d??h????9H?Z??t???_0?103O??@Kfm??<4?H?x'??8??.`????9??k(SV)?n?.?????S?0?? |???3b?0?"0??(`uH??????w?}?? ~?;??'?L+X????^/6???]*???w?????YC#6:?????`?&?.&???????J]b0????e???o???y????????ew
?????????????]~?*????w?/???M?????????|C????,???Y?????????????7?o???WAM????????0???S`?A??*+??L
?o???y????`/???-y?5q?l?N?c?L????k<?I6??'<???t"???x3?????r29??l? ? ?????????P??[?   IDAT?2?J?
1??x?.f?X.???? h???[YL'????T?[-L4?????X??_?n??G?4??????<;?xa???? <;???<??????mmm6?[+?mcq???``p??????'?%??4????S1???????I?????5?dE\????5%??????e?eA???5??|?,??R?_??N?/?_??/??????>:?	??;]?w??d??*?)?fAc*~;-?/a??;+)?????u?+????????4
?nE?3??m??93;???c?d?U??<,?W4hX?;???g?:?\?6Y?W?j???z?RD???????B8?o??_??V}r%?9f???~cR??j=E??7?k?Wx?D?2+8?`?g???H????6[???~????1[?r???q?????a?=????????????F??CIpg??n?U)??g????2??D`e
8???n??	?i% v?ye???$'?~ew?~?V???_??J???,?X?y??E??&?8?j?????)??b^1???[??????? +????8????oFP?U5?`????..???'a?yXEh?,???? ????????^?p?
?n?%?,?5?q??0?+?d???9UDS??k???w???@
?$?f??????rH?N??e(?L?d???????;??/E?`T?1'?$??heo???n?,???s??*???? ,&????E??`?U?-?iEKdf????>???I???]V?H?????????K?3? C?`*?BU?????x5D????^??????????????o??K?o??? ?l???^?o??oX??????H???Br.B_?:??*?`?Q`??&?r?"?TV??uD &?zbs&#c??[?/kq????]s???O ???K?? ????+??r/&>\?{?????????2SZRp??p??U?\?_|??U??/_^!^?<?????r?b7??on@??WW?!Rz??y?? l????2:[e??% ?3?????UFl?C|??? ???????t????0?
5?QmN?9????#??<????aUvAX ?%G3?EOi???]?>???g??}]?O???NW?? ?!?`P??["z??5i?J??I	u?T?k?F4@&te]g)???,t??5T?vnv???/??? 1????C_???????O????N?%bfzp?9???b?/S????????1:]?xt?)????	BQ?&??8?Jhj L?????N?I???*7T???,?? ????????N9?l?)?+??e?:$q>}Q:>?????`?~z(,??L?????Rm-?c??\Zns???^0p?d?4^mq???e??^wK?'?3?????_??V????e`??n?;??????N?;K?Ux??<?`?:?%???:!????Xw9f?X?tQQ???_??????????h?D_?E?|??*?w3 Nn????L??{?)?`r??X+?Zd?7#Xj?)?|???{D?? ?c????%?I???A|????????-&???l?l??-??,N?? ??s!???s ~(?9???[?G0??V?p?a?????/T?B??~oNYD/?/??e???)?gwv?<zC??????b??????2`]?mV??z??? ????<. ?h???P??H&? ???e???e?i?????N?&?fA????!??*???0`B?1	?5?P??8_V&t*??????c???g??<??9?n}??e!?? ;?^??}?W??;C_>)?`W?9,???e????4(?g?o??6O?pU??r??z???????,???W94??b??(4?????0?x2
??uE?XF?+???-?$?rc?????V}?~e?U]??uJ????????D|?>?t??????_????{)o???`p?8D?i??/?&?Q???eF? YK?_??U!+??o??Vew????????&7?Mv????-?[s???^!??o?~???N8??[ga@??#+:Q?3S?s?:a)??^???&|?????*?>D?#MN?o?		?u??L
?p??Q `???e??C?'!??K?`?ShT???;r???bw????y???H?+)X????`??}eA ??????jov?w;{ ?nw{?"?$& .
z-B?A?#/-K?X^????B???}???`??'??+20O?????	???????T????Z?_?M???31>????_??c??????|????o~?2?'?????????zv????n?f?[????&???3#3???3? `&??~)+?D????I?V?m?{?&X??