# coding=utf-8
# Author: spetrel@gmail.com

ROPE_SCALING = "rope_scaling"


def set_default(config: dict, key: str, default_value):
    if config.get(key, None) is None:
        config[key] = default_value


def get_quant_method(model_config: dict):
    quant_cfg = model_config.get("quantization_config", {})
    return quant_cfg.get("quant_method") if quant_cfg else None


def set_neox_style(model_config: dict, neox_style: bool):
    set_default(model_config, ROPE_SCALING, {})
    model_config[ROPE_SCALING]["neox_style"] = bool(neox_style)
