# coding=utf-8

import torch
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
        is_h20 = 'NVIDIA H20' in torch.cuda.get_device_name(0)
        set_env(LATENT_CACHE, 1)
        set_env("FREEZE_MEM_EACH_LAYER", 1)
        set_env("MOE_EXP_PARALLEL", 1)
        set_env("MOE_DYN_SHARED", 1)
        if is_h20:
            set_env('USE_FLASH_MLA', 2, "Turn on FlashMLA for H20")
            # set_env('ATTN_DATA_PARALLEL', 1)
            set_env('ATTN_DATA_PARALLEL_MIN_BATCH', 4)
        if get_int_env('USE_FLASH_MLA') > 0:
            set_env("KV_CACHE_ALIGN_PAGE", 64)
            set_env(PRE_ALLOC_ALL_TOKEN, 1)

        set_default(config, "norm_topk_prob", False)
        if config.get("use_mla", True):
            set_neox_style(config, False)
        set_default(config, "topk_method", "greedy")
        set_default(config, "scoring_func", "softmax")
        assert config["topk_method"] in ("greedy", "group_limited_greedy", "noaux_tc")
        assert config["scoring_func"] in ("softmax", "sigmoid")
        # Reverse more for big model (MOE)
        work_mem_extra = int(config.get("num_experts_per_tok", 8)) * int(config.get("hidden_size", 7168))
        set_env("reserved_work_mem_mb", 1500 + work_mem_extra * 20 // 1000)

        # quant config
        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")
        awq_use_exllama(config)
        if quant_method == "gptq" or quant_method == "awq" and get_int_env("AWQ_USE_EXLLAMA") == 1:
            set_env("FUSE_GPTQ_MOE", 1)
            if config.get("torch_dtype", "") == "bfloat16" and "force_half" not in config:
                print("WARNING: force convert to half dtype for using GPTQ kernel")
                config["bfloat16"] = False
                config["force_half"] = True


class DeepseekV3Adapter:
    @staticmethod
    def adapt(config: dict):
        DeepseekV2Adapter.adapt(config)
