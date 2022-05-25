import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .kd_one_stage import KnowledgeDistillationSingleStageDetector


@DETECTORS.register_module()
class LAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_ckpt,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KnowledgeDistillationSingleStageDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)
        self.eval_teacher = eval_teacher
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = build_head(teacher_bbox_head)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
 