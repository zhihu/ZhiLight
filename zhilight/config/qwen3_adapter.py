# coding=utf-8

import torch
from .config_util import *
from .dev_config import *


def _is_fp8_block(quant_config: dict):
    return quant_config.get("quant_method", "") == "fp8" and "weight_block_size" in quant_config


class Qwen3Adapter:
    @staticmethod
    def adapt(config: dict):
        if config["num_hidden_layers"] == 64 and config["hidden_size"] == 5120:
            m_size = '32b'
            print(f"##### Adapt qwen3 {m_size} config ########")
            set_envs({
                HIGH_PRECISION: 0,
                FUSE_QKV: 1,
                FUSE_FF_IN: 2,
                DUAL_STREAM: 1,
                REDUCE_TP_INT8_THRES: 1000,
            })
            if get_quant_method(config) == "awq":
                set_env("AWQ_USE_EXLLAMA", 1)


class Qwen3MOEAdapter:
    @staticmethod
    def adapt(config: dict):
        set_env("FREEZE_MEM_EACH_LAYER", 1)
        set_env("MOE_EXP_PARALLEL", 1)
        set_env("MOE_ROUTER_FLOAT", 1)
        set_env("EMBEDDING_AUTO_CAST", 1)  # TODO: test the effect
        set_env(FUSE_QKV, 1)

        set_default(config, "use_qk_norm", True)

        set_default(config, "norm_topk_prob", False)
        set_default(config, "topk_method", "greedy")
        set_default(config, "scoring_func", "softmax")

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

        # Reverse more work memory for big model (MOE)
        work_mem_base = 1500  # MB
        if get_int_env("GROUPED_FP8_GEMM") > 0:
            work_mem_base = 1800
        work_mem_extra = int(config.get("num_experts_per_tok", 8)) * int(config.get("hidden_size", 4096))
        set_env("reserved_work_mem_mb", work_mem_base + work_mem_extra * 20 // 1000)
