# coding=utf-8

from .dev_config import *


class CohereAdapter:
    @staticmethod
    def adapt(config: dict):
        config["eps"] = config["layer_norm_eps"]
        if 'tie_lm_head' not in config:
            config["tie_lm_head"] = True
        if config.get("use_qk_norm", False):
            os.environ[FUSE_QKV] = "0"  # can't fuse because qk norm
        os.environ[DUAL_STREAM] = "0"  # EncoderLayer is different from LlaMA
        os.environ["DEQUANT_DESC_ACT"] = "1"  # dequant attn_out to speedup
