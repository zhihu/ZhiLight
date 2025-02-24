# coding=utf-8

from .config_util import *
from .dev_config import *


def awq_use_exllama(config: dict):
    quant_method = get_quant_method(config)
    if quant_method == "awq":
        print("Use Int4GPTQ for AWQ checkpoint.")  # Linear::Int4GPTQ
        set_env("AWQ_USE_EXLLAMA", 1)  # Linear::AWQ::get_dequant_weight() is not implemented


class DeepseekV2Adapter:
    @staticmethod
    def adapt(config: dict):
        set_env(LATENT_CACHE, 1)
        set_env("FREEZE_MEM_EACH_LAYER", 1)
        set_env("MOE_EXP_PARALLEL", 1)
        set_env("MOE_DYN_SHARED", 1)

        set_default(config, "norm_topk_prob", False)
        if config.get("use_mla", True):
            set_neox_style(config, False)
        set_default(config, "topk_method", "greedy")
        set_default(config, "scoring_func", "softmax")
        assert config["topk_method"] in ("greedy", "group_limited_greedy", "noaux_tc")
        assert config["scoring_func"] in ("softmax", "sigmoid")

        # quant config
        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")
        awq_use_exllama(config)
        if quant_method:
            set_env("FUSE_GPTQ_MOE", 1)
            if config.get("torch_dtype", "") == "bfloat16" and "force_half" not in config:
                print("WARNING: force convert to half dtype for using GPTQ kernel")
                config["bfloat16"] = False
                config["force_half"] = True


class DeepseekV3Adapter:
    @staticmethod
    def adapt(config: dict):
        DeepseekV2Adapter.adapt(config)
