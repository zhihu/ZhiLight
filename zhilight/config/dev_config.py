# coding=utf-8
# Author: spetrel@gmail.com

import os

# Developing configurations
# most of them only have impact on the speed.

""" All Reduce """
# Turn on dual streams optimization for prefill on GPUs connected by PCI-e.
# This param has no effect on A100/A800.
DUAL_STREAM = "DUAL_STREAM"
DUAL_STREAM_THRESHOLD = "DUAL_STREAM_THRESHOLD"
# Turn of int8 quantization for all reduce if the prompt's length bigger than this threshold. Default: off
REDUCE_TP_INT8_THRES = "REDUCE_TP_INT8_THRES"
HOST_REDUCE = "HOST_REDUCE"  # Turn on host CPU reducing. It's faster when the number of GPUs is 2.
HOST_REDUCE_COPY_ONLY = "HOST_REDUCE_COPY_ONLY"

""" Gemm """
HIGH_PRECISION = "HIGH_PRECISION"  # GEMM accumulator option: 1: float32; 0: float16/bfloat16
FUSE_QKV = "CPM_FUSE_QKV"  # Fuse attention Q,K,V three matrices GEMM into one
FUSE_FF_IN = "CPM_FUSE_FF_IN"  # Fuse Feedforward 'in', 'gate; two matrices GEMM into one

CHUNKED_PREFILL = "CHUNKED_PREFILL"
CHUNKED_PREFILL_SIZE = "CHUNKED_PREFILL_SIZE"

""" KV Cache """
PRE_ALLOC_ALL_TOKEN = "PRE_ALLOC_ALL_TOKEN"
LATENT_CACHE = "LATENT_CACHE"  # for MLA

""" Memory """
RESERVE_MEM_MB = "RESERVE_MEM_MB"  # Reserve GPU memory NOT used by ZhiLight

ROPE_CACHE = "ROPE_CACHE"


def get_int_env(name, default=0):
    return int(os.environ.get(name, default))


def set_env(name, value, tip=''):
    if name not in os.environ:
        print(f"### Auto Set dev env: {name}={value} {tip}")
        os.environ[name] = str(value)


def set_env_kv(**kwargs):
    for k, v in kwargs.items():
        set_env(k, str(v))


def set_envs(env_dict: dict):
    for k, v in env_dict.items():
        set_env(k, v)


def get_quant_method(model_config: dict):
    quant_cfg = model_config.get("quantization_config", {})
    return quant_cfg.get("quant_method") if quant_cfg else None
