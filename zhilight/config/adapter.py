# coding=utf-8
# Author: spetrel@gmail.com

from .dev_config import *
from .config_util import *
# Keep in alphabetical order
from .cohere_adapter import CohereAdapter
from .deepseek_adapter import DeepseekV2Adapter, DeepseekV3Adapter
from .llama_adapter import LLaMAAdapter
from .qwen2_adapter import Qwen2Adapter
from .qwen3_adapter import Qwen3Adapter, Qwen3MOEAdapter


def _get_model_type(config: dict):
    model_type = config.get("model_type", "")
    if model_type:
        return model_type
    architectures = config.get("architectures", [""])
    if "minicpm" in architectures[0].lower():
        config["model_type"] = "cpm_dragonfly"
        return "cpm_dragonfly"
    return ""


class ModelAdapter:
    @staticmethod
    def adapt(config: dict):
        model_type = config.get("model_type", "")
        set_default(config, "rope_scaling", {})

        # Keep in alphabetical order
        if model_type == "cohere":
            CohereAdapter.adapt(config)
        elif model_type == "deepseek_v2":
            DeepseekV2Adapter.adapt(config)
        elif model_type == "deepseek_v3":
            DeepseekV3Adapter.adapt(config)
        elif model_type == "llama":
            LLaMAAdapter.adapt(config)
        elif model_type == "qwen2":
            Qwen2Adapter.adapt(config)
        elif model_type == "qwen3":
            Qwen3Adapter.adapt(config)
        elif model_type == "qwen3_moe":
            Qwen3MOEAdapter.adapt(config)

        if get_int_env(CHUNKED_PREFILL) == 1:
            set_env("DUAL_STREAM_THRESHOLD", 100)

        ModelAdapter.adapt_gptq(config)

        return config

    @staticmethod
    def adapt_gptq(config: dict):
        quant_config = config.get("quantization_config", {})
        if (
                quant_config
                and quant_config.get("desc_act", False)
                and config.get("bfloat16", False)
                and not config.get("force_half", False)
        ):
            print("WARNING: force convert to half dtype for using GPTQ kernel")
            config["bfloat16"] = False
            config["force_half"] = True
