# coding=utf-8

import glob
import json
import numpy as np
import os
import random
import re
import sys
import time
from . import C
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
    Callable,
    NamedTuple,
)
from transformers import AutoTokenizer
from typing_extensions import TypedDict
from .config.adapter import ModelAdapter
from .config.dist_config import DistConfig
from .loader import ModelLoader, LLaMALoader
from .quant import QuantConfig, QuantType, quant_config_to_c

LLaMAInputType = Union[str, List[str]]


class LLaMAInferenceResults(TypedDict, total=False):
    result: Union[LLaMAInputType, List[LLaMAInputType]]
    score: float
    time_elapsed: float
    input_tokens_num: int
    output_tokens_num: int


class LLaMAModelConfig(TypedDict):
    num_layers: int
    dim_model: int
    num_heads: int
    dim_head: int
    dim_ff: int
    vocab_size: int
    eps: float
    num_kv_heads: int
    new_vocab: bool
    bfloat: bool
    pos_bias_type: str
    activate_fn: str
    scale_weights: bool
    weight_transposed: bool
    max_token: int
    tie_lm_head: bool
    rope_theta: float


def _get_config(model_config) -> LLaMAModelConfig:
    cfg = {
        "num_layers": 32,
        "dim_model": 4096,
        "num_heads": 32,
        "dim_ff": 11008,
        "vocab_size": 32000,
        "eps": 1e-6,
        "pos_bias_type": "rotary",
        "activate_fn": "silu",
        "scale_weights": False,
        "weight_transposed": False,
        "max_token": 4096,
        "tie_lm_head": False,
        "rope_theta": 10000.0,
    }
    if model_config is not None:
        model_type = model_config.get("model_type", "cpm_caterpillar")
        if model_type == "cpm_dragonfly":
            cfg["dim_head"] = 64
            cfg["tie_lm_head"] = True
            if "num_experts" in model_config:
                cfg["moe_num_experts"] = model_config["num_experts"]
        cfg.update(model_config)
        if model_type == "cpm_caterpillar" and "scale" in model_config:
            cfg["scale_weights"] = model_config["scale"]
    if cfg.get("_dtype", "") == "bf16":
        cfg["bfloat16"] = True
    if "tie_word_embeddings" in cfg:
        cfg["tie_lm_head"] = cfg.get("tie_word_embeddings", False)
    # cfg["eps"] = 1e-6
    if not "quantization_config" in cfg:  # disable fuse for GPTQ temporarily
        if not os.environ.get('CPM_FUSE_QKV'):
            os.environ["CPM_FUSE_QKV"] = "1"
        if not os.environ.get('CPM_FUSE_FF_IN'):
            os.environ["CPM_FUSE_FF_IN"] = "1"
    if not os.environ.get('HIGH_PRECISION'):
        # Compute_32F
        os.environ["HIGH_PRECISION"] = "1"
    if not os.environ.get('W4_A8_M_THRES'):
        os.environ["W4_A8_M_THRES"] = "1000"
    print(f'[DEV]Config: '
          f'HIGH_PRECISION={os.environ.get("HIGH_PRECISION")}'
          f'; DUAL_STREAM={os.environ.get("DUAL_STREAM")}'
          f'; CPM_FUSE_QKV={os.environ.get("CPM_FUSE_QKV")}'
          f'; CPM_FUSE_FF_IN={os.environ.get("CPM_FUSE_FF_IN")}'
          f'; REDUCE_TP_INT8_THRES={os.environ.get("REDUCE_TP_INT8_THRES")}'
          f'; W4_INT8_ALGO={os.environ.get("W4_INT8_ALGO")}'
          f'; W4_FP8_ALGO={os.environ.get("W4_FP8_ALGO")}'
          )
    return cfg


class LLaMA:
    def __init__(
        self,
        model_path: str,
        model_config: Optional[LLaMAModelConfig] = None,
        quant_config: Optional[QuantConfig] = None,
        dist_config: Optional[DistConfig] = None,
        tokenizer=None,
        device_id: int = -1,  # deprecated
        memory_limit: int = 0,  # deprecated
        **kwargs
    ):
        self._config = _get_config(ModelAdapter.adapt(model_config))
        self._quant_config = QuantConfig.adapt_hf_config(quant_config, self._config)

        self._init_tokenizer(model_path, tokenizer)

        c_config = C.ModelConfig(self._config)
        c_quant_config = quant_config_to_c(self._quant_config)
        if dist_config is None:
            dist_config = DistConfig()
        c_dist_config = dist_config.to_c_config()

        self._context = None  # Reserve for future usage

        c_engine = C.Engine(device_id, memory_limit, c_dist_config)

        self._model = C.LLaMA(
            c_engine,
            c_config,
            c_quant_config,
            c_dist_config,
        )

    def process_inputs(self, messages: List[dict]):
        debug_feed_embedding = False
        if debug_feed_embedding:
            print("#################### process_inputs")
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            positions = list(range(0, len(token_ids)))
            inputs_embeds = self._model.get_input_embeddings(token_ids)
            # TODO: define a class instead of returning dict
            return (token_ids,
                    positions,
                    inputs_embeds,
                    None,
                    )
        return None

    def _load_tokenizer_config(self, model_path: str):
        tokenizer_config_path = f"{model_path}/tokenizer_config.json"
        if os.path.exists(tokenizer_config_path):
            self.tokenizer_config = json.load(open(tokenizer_config_path))
        else:
            self.tokenizer_config = {}

    def _init_tokenizer(self, model_path: str, tokenizer):
        self._load_tokenizer_config(model_path)
        if tokenizer is not None:
            self._tokenizer = tokenizer
            return

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        self._tokenizer.bos_token_id = self._config.get("bos_token_id", 1)
        conf_eos_token_id = self._config.get("eos_token_id", 2)
        if isinstance(conf_eos_token_id, int):
            self._tokenizer.eos_token_id = conf_eos_token_id
        elif self._tokenizer.eos_token_id is None:
            self._tokenizer.eos_token_id = 2

    def load_state_dict_pt(self, state_dict):
        import torch
        def trans_type(dtype, p):
            if dtype == torch.bfloat16:
                return p.view(torch.int16)
            if dtype == torch.float8_e4m3fn:
                return p.view(torch.int8)
            if dtype == torch.float32 and p.ndim > 0:
                return p.half()
            return p

        if self._config.get("force_half", False):
            for name, param in list(state_dict.items()):
                if param.dtype == torch.bfloat16:
                    state_dict[name] = param.half()

        new_state_dict = {
            LLaMALoader._replace_name(name): np.atleast_1d(trans_type(param.dtype, param).cpu().numpy())
            for name, param in state_dict.items()
        }
        self._model.load_state_dict(new_state_dict)

    def load_model_pt(self, model_dir):
        state_dict = LLaMALoader.load_pt(model_dir)
        self.load_state_dict_pt(state_dict)

    def load_model_safetensors(self, model_dir, pattern="*.safetensors"):
        self.load_state_dict_pt(LLaMALoader.load_safetensors(model_dir, pattern))

    def load_with_smooth_quant(
        self,
        model_dir_or_state_dict: Union[str, dict],
        act_scale_pt: str,
        alpha: float = 0.5,
        min_scale: float = 1e-5,
        max_scale: float = 1e5,
    ):
        import torch

        if isinstance(model_dir_or_state_dict, str):
            state_dict = LLaMALoader.load_pt(model_dir_or_state_dict)
        else:
            state_dict: dict = model_dir_or_state_dict
        new_state_dict = {}
        for name, p in state_dict.items():
            name = LLaMALoader._replace_name(name)
            new_state_dict[name] = p.numpy() if isinstance(p, torch.Tensor) else p
        act_scale_dict = torch.load(act_scale_pt, map_location="cpu")
        act_scale_dict = {name: p.half().numpy() for name, p in act_scale_dict.items()}
        self._model.load_with_smooth_quant(
            new_state_dict, act_scale_dict, alpha, min_scale, max_scale
        )

    # for smooth quant
    def calc_act_scales(self, data: List[LLaMAInputType]):
        return self._model.calc_act_scales([self._tokenizer.encode(d) for d in data])

    def get_tokens(self, text: str):
        ids = self._tokenizer.encode(text, with_bos=False)
        return self._tokenizer.decode(ids, skip_special_tokens=True, join_tokens=False)

