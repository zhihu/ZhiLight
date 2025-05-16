# coding=utf-8

import torch
from .config_util import *
from .dev_config import *


def _is_fp8_block(quant_config: dict):
    return quant_config.get("quant_method", "") == "fp8" and "weight_block_size" in quant_config


def _is_671b(config: dict):
    return config["max_position_embeddings"] == 163840 and config["num_hidden_layers"] == 61


class DeepseekV2Adapter:
    @staticmethod
    def adapt(config: dict):
        is_h20 = 'NVIDIA H20' in torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        set_env(LATENT_CACHE, 1)
        set_env("FREEZE_MEM_EACH_LAYER", 1)
        set_env("MOE_EXP_PARALLEL", 1)
        set_env("MOE_DYN_SHARED", 1)
        if is_h20:
            set_env('USE_FLASH_MLA', 2, "Turn on FlashMLA for H20")
            # set_env('ATTN_DATA_PARALLEL', 1)
            set_env('ATTN_DATA_PARALLEL_MIN_BATCH', 0)
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

        # quant config
        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")
        awq_use_exllama(quant_config)
        if quant_method == "gptq" or quant_method == "awq" and get_int_env("AWQ_USE_EXLLAMA") == 1:
            set_env("FUSE_GPTQ_MOE", 1)
            if config.get("torch_dtype", "") == "bfloat16" and "force_half" not in config:
                print("WARNING: force convert to half dtype for using GPTQ kernel")
                config["bfloat16"] = False
                config["force_half"] = True
        if _is_fp8_block(quant_config):
            set_env("GROUPED_FP8_GEMM", 1)
        if _is_fp8_block(quant_config) and _is_671b(config) and total_memory < 100 * (2 ** 30):
            # H20 96G isn't enough to hold all replicated weights for MLA layer for DeepSeek R1,
            # So we keep o_proj in TP mode to reduce memory usage.
            set_env('ATTN_DATA_PARALLEL_OUT', 0, "Prevent OOM")
        # Reverse more work memory for big model (MOE)
        work_mem_base = 1500  # MB
        if get_int_env("GROUPED_FP8_GEMM") > 0:
            work_mem_base = 1800
        work_mem_extra = int(config.get("num_experts_per_tok", 8)) * int(config.get("hidden_size", 7168))
        set_env("reserved_work_mem_mb", work_mem_base + work_mem_extra * 20 // 1000)


class DeepseekV3Adapter:
    @staticmethod
    def adapt(config: dict):
        DeepseekV2Adapter.adapt(config)
