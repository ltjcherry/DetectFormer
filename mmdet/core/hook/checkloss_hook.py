# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


class BaseEMAHook(Hook):
    """Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_para