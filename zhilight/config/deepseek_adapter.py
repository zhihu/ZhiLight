# coding=utf-8

from .config_util import *
from .dev_config import *


class DeepseekV2Adapter:
    @staticmethod
    def adapt(config: dict):
        set_env(LATENT_CACHE, 1)
        set_default(config, "norm_topk_prob", False)
        if config.get("use_mla", True):
            set_neox_style(config, False)
