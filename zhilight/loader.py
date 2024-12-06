# coding=utf-8

import concurrent.futures
import os
import re
import glob
import json
import time
import torch
from typing import Dict
from abc import ABC, abstractmethod
from .quant import QuantConfig, QuantType


class ModelLoader(ABC):
    @abstractmethod
    def fetch_parameter(self, name):
        ...

    @property
    @abstractmethod
    def model_config(self):
        ...

    @property
    @abstractmethod
    def quant_config(self):
        ...

    @property
    @abstractmethod
    def vocab_path(self):
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        ...

    @staticmethod
    def convert_pt_to_safetensors(model_dir, pattern="*.pt"):
        from safetensors.torch import save_file
        files = glob.glob(f"{model_dir}/{pattern}")
        for f in files:
            print(f"Convert {f}")
            state_dict = torch.load(f, "cpu")
            # state_dict = ModelLoader.load_pt(f)
            save_file(state_dict, f.split('.', 2)[0] + ".safetensors")
            print(f"Done Convert {f}")

    @staticmethod
    def load_safetensors(model_dir, pattern="*.safetensors", parallel=None):
        if not hasattr(torch, 'float8_e4m3fn'):
            torch.float8_e4m3fn = torch.int8
        from safetensors.torch import load_file
        files = sorted(glob.glob(f"{model_dir}/{pattern}"))
        if not files:
            raise ValueError(f"No safetensors files found in: {model_dir}")
        state_dict = {}
        if parallel is None:
            parallel = model_dir.startswith("/mnt") and os.environ.get("DISABLE_PARALLEL_LOAD", "0") != "1"
        if parallel:
            print(f"########## parallel load_clone {len(files)} files ##########")
            t0 = time.time()
            def load_clone(f):
                d1 = load_file(f)
                return {k: torch.clone(v) for k, v in d1.items()}
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(load_clone, f) for f in files]
                for f in futures:
                    state_dict.update(f.result())
            print(f"########## Done load_clone {len(files)} files in {time.time() - t0:.1f} seconds ##########")
            return state_dict
        for f in files:
            state_dict.update(load_file(f))
        return state_dict

    @staticmethod
    def load_pt(model_dir):
        state_dict = {}
        if os.path.isfile(f"{model_dir}"):
            state_dict = torch.load(f"{model_dir}", map_location="cpu")
        elif os.path.isfile(f"{model_dir}/pytorch_model.pt"):
            state_dict = torch.load(f"{model_dir}/pytorch_model.pt", map_location="cpu")
        else:
            pt_files = sorted(glob.glob(f"{model_dir}/pytorch_model*.bin"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/caterpillar_*.pt"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/cpm_*.pt"))
            if not pt_files and glob.glob(f"{model_dir}/*.safetensors"):
                return ModelLoader.load_safetensors(model_dir)
            if not pt_files:
                raise ValueError(f"No checkpoint found in: {model_dir}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(torch.load, f, "cpu") for f in pt_files]
                for f in futures:
                    state_dict.update(f.result())

        return state_dict

    @staticmethod
    def _lazy_load_model_pt(model_dir):
        from zhilight.lazy_unpickling import LazyUnpickleStorage

        if os.path.isfile(f"{model_dir}"):
            state_dict = LazyUnpickleStorage(f"{model_dir}")
        elif os.path.isfile(f"{model_dir}/pytorch_model.pt"):
            state_dict = LazyUnpickleStorage(f"{model_dir}/pytorch_model.pt")
        else:
            pt_files = sorted(glob.glob(f"{model_dir}/pytorch_model*.bin"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/caterpillar_*.pt"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/cpm_*.pt"))
            if not pt_files:
                raise ValueError(f"No checkpoint found in: {model_dir}")
            elif len(pt_files) > 1:
                raise ValueError("lazy loading only support single pt file!")
            state_dict = LazyUnpickleStorage(pt_files[0])

        return state_dict


def _set_env(name, value, tip=''):
    if name not in os.environ:
        print(f"### Auto Set {name}={value} {tip}")
        os.environ[name] = str(value)


def _set_envs(env_dict: dict):
    for k, v in env_dict.items():
        _set_env(k, v)


def _get_quant_method(model_config: dict):
    quant_cfg = model_config.get("quantization_config", {})
    return quant_cfg.get("quant_method") if quant_cfg else None


def _adapt_qwen2_config(model_config: dict, ext_config: dict):
    if model_config["num_layers"] in [48, 28] and model_config["dim_model"] in [5120, 3584]:
        m_size = '14b' if model_config["num_layers"] == 48 else '7b'
        print(f"##### Adapt qwen2 {m_size} config ########")
        _set_envs({
            "CHUNKED_PREFILL": 1,
            "CHUNKED_PREFILL_SIZE": 512,
            "CPM_FUSE_QKV": 1,
            "CPM_FUSE_FF_IN": 1,
            "INT8_TRT": 1,
        })
        if os.environ.get("CHUNKED_PREFILL", "") == "1":
            _set_envs({
                "HOST_REDUCE": 1,
                "HOST_REDUCE_COPY_ONLY": 1,
                "DUAL_STREAM": 1,
                "DUAL_STREAM_THRESHOLD": 100,
            })
    if model_config["num_layers"] == 80 and model_config["dim_model"] == 8192:
        m_size = '72b' if model_config["intermediate_size"] == 29696 else '110b'
        print(f"##### Adapt qwen2 {m_size} config ########")
        _set_envs({
            "HIGH_PRECISION": 0,
            "CPM_FUSE_QKV": 1,
            "CPM_FUSE_FF_IN": 2,
            "DUAL_STREAM": 1,
            "REDUCE_TP_INT8_THRES": 1000,
        })
        if m_size == '110b':
            _set_env("PRE_ALLOC_ALL_TOKEN", 0, "for 110b to reduce memory usage.")
        if _get_quant_method(model_config) == "awq":
            _set_env("AWQ_USE_EXLLAMA", 1)
    if "rope_scaling" in model_config:
        if "factor" in model_config["rope_scaling"] and model_config["rope_scaling"]["factor"] > 1.:
            if os.environ.get("DISABLE_ROPE_SCALING", "") == "1":
                model_config.pop("rope_scaling")
            _set_env("CHUNKED_PREFILL", 1, "for LONG context to reduce memory usage!")
            _set_envs({"CHUNKED_PREFILL_SIZE": 8192})


class LLaMALoader(ModelLoader):
    def __init__(self, model_path: str, lazy: bool = False):
        self._model_path = model_path
        self._model_config = json.load(open(f"{model_path}/config.json"))
        if "new_vocab" not in self._model_config:
            self._model_config["new_vocab"] = False
        if "is_chatml" not in self._model_config:
            self._model_config["is_chatml"] = False
        if self._model_config.get("_dtype", "") == "bf16":
            self._model_config["bfloat16"] = True

        self._model_config["weight_transposed"] = False
        self._state_dict = (
            self._lazy_load_model_pt(model_path)
            if lazy
            else self.load_pt(model_path)
        )
        self._name_mapping = {
            self._replace_name(name): name for name in self._state_dict.keys()
        }
        self._vocab_path = f"{model_path}/vocabs.txt"
        self._tokenizer = None

    @staticmethod
    def load_llama_config(model_path: str):
        if not os.path.isfile(f"{model_path}/config.json"):
            raise ValueError(f"{model_path}/config.json not found")
        with open(f"{model_path}/config.json") as fp:
            model_config = json.load(fp)
        return LLaMALoader.adapt_hf_config(model_config)

    @staticmethod
    def adapt_hf_config(model_config: dict):
        # if not os.path.isfile(f"{model_path}/config.json"):
        #     return model_config
        # with open(f"{model_path}/config.json") as fp:
        #     ext_config = json.load(fp)
        ext_config = model_config
        if "num_hidden_layers" in ext_config:
            # model_config.update({k: v for (k, v) in ext_config.items() if k not in model_config})
            model_config["num_layers"] = ext_config["num_hidden_layers"]
            model_config["dim_model"] = ext_config["hidden_size"]
            model_config["num_heads"] = ext_config["num_attention_heads"]
            if "num_key_value_heads" in ext_config:
                model_config["num_kv_heads"] = ext_config["num_key_value_heads"]
            if "max_position_embeddings" in ext_config:
                model_config["max_token"] = ext_config["max_position_embeddings"]
            model_config["dim_ff"] = ext_config["intermediate_size"]
            if "rms_norm_eps" in ext_config:
                model_config["eps"] = ext_config["rms_norm_eps"]
            model_config["activate_fn"] = ext_config["hidden_act"]
            model_config["bfloat16"] = "bfloat16" == ext_config["torch_dtype"]
            model_config["new_vocab"] = False

        if ext_config.get("model_type", "") == "qwen2":
            _adapt_qwen2_config(model_config, ext_config)

        if ext_config.get("model_type", "") == "deepseek_v2":
            _set_env("FREEZE_MEM_EACH_LAYER", 1)
            _set_env("LATENT_CACHE", 1)
            _set_env("FUSE_ATTN_SEARCH", 0)
            _set_env("CPM_FUSE_FF_IN", 2)
            _set_env("MOE_EXP_PARALLEL", 1)
            quant_cfg = ext_config.get("quantization_config", {})
            quant_method = None
            if quant_cfg:
                quant_method = quant_cfg.get("quant_method")
                if quant_method == "awq":
                    _set_env("AWQ_USE_EXLLAMA", 1)
                if quant_method == "awq" or quant_method == "gptq":
                    _set_env("FUSE_GPTQ_MOE", 1)
                    _set_env("MOE_DYN_SHARED", 1)
                    _set_env("GPTQ_MOE_M_THRES", 16)
                    _set_env("ROUTED_SCALING_TO_WEIGHT", 1)  # prevent float16 underflow
            if quant_method == 'awq' and ext_config["intermediate_size"] == 10944:
                print(f"##### Fix intermediate_size for TP!!! #####")
                ext_config["intermediate_size"] += 64
            s = int(ext_config["n_shared_experts"]) * int(ext_config["moe_intermediate_size"])
            model_config["shared_expert_intermediate_size"] = s
            dim_head = ext_config["qk_nope_head_dim"] + ext_config["qk_rope_head_dim"]
            model_config["dim_head"] = dim_head

        if ext_config.get("model_type", "") == "cohere":
            model_config["eps"] = ext_config["layer_norm_eps"]
            if 'tie_lm_head' not in ext_config:
                model_config["tie_lm_head"] = True
            if ext_config.get("use_qk_norm", False):
                os.environ["CPM_FUSE_QKV"] = "0"
            os.environ["DUAL_STREAM"] = "0"  # EncoderLayer is different
            os.environ["DEQUANT_DESC_ACT"] = "1"  # dequant attn_out to speedup

        if ext_config.get("model_type", None) is None and ext_config["architectures"][0].lower().startswith("minicpm"):
            model_config["model_type"] = "cpm_dragonfly"

        if os.environ.get("CHUNKED_PREFILL", "") == "1":
            _set_env("DUAL_STREAM_THRESHOLD", 100)

        quant_config = ext_config.get("quantization_config", {})
        if (
            quant_config
            and quant_config.get("desc_act", False)
            and model_config.get("bfloat16", False)
            and not model_config.get("force_half", False)
        ):
            print("WARNING: force convert to half dtype for using GPTQ kernel")
            model_config["bfloat16"] = False
            model_config["force_half"] = True
        return model_config

    def fetch_parameter(self, name):
        try:
            orig_name = self._name_mapping[name]
            tensor = self._state_dict[orig_name]
            return (
                tensor.numpy()
                if tensor.dtype != torch.bfloat16
                else tensor.view(torch.int16).numpy()
            )
        except Exception as e:
            print("err", e)
            raise e

    @property
    def model_config(self) -> Dict:
        return self._model_config

    @property
    def quant_config(self) -> Dict:
        return {"type": QuantType.NoQuant}

    @property
    def vocab_path(self):
        return self._vocab_path

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import LlamaTokenizer
            self._tokenizer = LlamaTokenizer.from_pretrained(self._model_path)
            self._tokenizer.bos_token_id = 1
            self._tokenizer.eos_token_id = 2
            print("vocab_size:", self._tokenizer.vocab_size)
        return self._tokenizer

    @staticmethod
    def convert_quant_dict(state_dict):
        if 'quant_state' not in state_dict:
            return state_dict
        state = state_dict['state']
        quant_state = state_dict['quant_state']
        state_dict = {}
        for name, param in state.items():
            if name in quant_state:
                prefix = name.rsplit('.', 1)[0]
                state_dict[prefix + ".quant_weight"] = param
                state_dict[prefix + ".scale_weight"] = quant_state[name]['scales']
                state_dict[prefix + ".zero_weight"] = quant_state[name]['qzeros']
            elif isinstance(param, torch.Tensor):
                state_dict[name] = param
        return state_dict

    @staticmethod
    def load_safetensors(model_dir, pattern="*.safetensors"):
        state_dict = ModelLoader.load_safetensors(model_dir, pattern)

        def _extend(name, x, dim=1):
            a = state_dict[name]
            shape = [d for d in a.shape]
            shape[dim] += x
            b = torch.zeros(*shape, dtype=a.dtype)
            if dim == 0:
                b[:a.shape[0], :] = a
            else:
                b[:, :a.shape[1]] = a
            state_dict[name] = b

        if 'model.layers.0.mlp.down_proj.qweight' in state_dict and \
                state_dict['model.layers.0.mlp.down_proj.qweight'].shape[0] == 10944:
            _extend('model.layers.0.mlp.up_proj.qweight', 8)
            _extend('model.layers.0.mlp.up_proj.qzeros', 8)
            _extend('model.layers.0.mlp.up_proj.scales', 64)
            _extend('model.layers.0.mlp.gate_proj.qweight', 8)
            _extend('model.layers.0.mlp.gate_proj.qzeros', 8)
            _extend('model.layers.0.mlp.gate_proj.scales', 64)
            _extend('model.layers.0.mlp.down_proj.qweight', 64, dim=0)
            _extend('model.layers.0.mlp.down_proj.qzeros', 1, dim=0)
            _extend('model.layers.0.mlp.down_proj.scales', 1, dim=0)

        return LLaMALoader.convert_quant_dict(state_dict)

    @staticmethod
    def load_pt(model_dir):
        state_dict = ModelLoader.load_pt(model_dir)
        return LLaMALoader.convert_quant_dict(state_dict)

    @staticmethod
    def _replace_name(s):
        s = re.sub("model.embed_tokens.weight", "token_embedding.weight", s)
        s = re.sub("model.norm.weight", "output_layernorm.weight", s)
        s = re.sub(
            "model.layers.([0-9]+).input_layernorm.(weight|scales|qweight|qzeros)",
            "layers.\\1.ln_attn.\\2",
            s,
        )
        s = re.sub(
            "model.layers.([0-9]+).post_attention_layernorm.weight",
            "layers.\\1.ln_ff.weight",
            s,
        )
        s = re.sub(
            "model.layers.([0-9]+).self_attn.([qkv])_proj.",
            "layers.\\1.attn.project_\\2.",
            s,
        )
        s = re.sub(
            "model.layers.([0-9]+).self_attn.o_proj.",
            "layers.\\1.attn.attn_out.",
            s,
        )
        s = re.sub("model.layers.([0-9]+).mlp.gate_proj.", "layers.\\1.ff.w_in.", s)
        s = re.sub("model.layers.([0-9]+).mlp.up_proj.", "layers.\\1.ff.w_gated.", s)
        s = re.sub("model.layers.([0-9]+).mlp.down_proj.", "layers.\\1.ff.w_out.", s)
        # llama2
        s = re.sub("input_embedding.weight", "token_embedding.weight", s)
        s = re.sub("encoder.output_layernorm.weight", "output_layernorm.weight", s)
        s = re.sub("output_projection.weight", "lm_head.weight", s)
        s = re.sub(
            "encoder.layers.([0-9]+).self_att.layernorm_before_attention.weight",
            "layers.\\1.ln_attn.weight",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).self_att.self_attention.project_([qkv]).(.*)(weight|qweight|scales|qzeros|g_idx)",
            "layers.\\1.attn.project_\\2.\\3\\4",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).self_att.self_attention.attention_out.(.*)(weight|qweight|scales|qzeros|g_idx)",
            "layers.\\1.attn.attn_out.\\2\\3",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).ffn.layernorm_before_ffn.weight",
            "layers.\\1.ln_ff.weight",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).ffn.ffn.w_in.w_0.(.*)(weight|qweight|scales|qzeros|g_idx)",
            "layers.\\1.ff.w_in.\\2\\3",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).ffn.ffn.w_in.w_1.(.*)(weight|qweight|scales|qzeros|g_idx)",
            "layers.\\1.ff.w_gated.\\2\\3",
            s,
        )
        s = re.sub(
            "encoder.layers.([0-9]+).ffn.ffn.w_out.(.*)(weight|qweight|scales|qzeros|g_idx)",
            "layers.\\1.ff.w_out.\\2\\3",
            s,
        )

        # Mixtral
        if "block_sparse_moe" in s or "mlp" in s:
            s = s.replace("shared_experts", "shared_expert")
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).experts.(\\d+).(w1|gate_proj).",
                       "layers.\\1.ff.experts.\\3.w_in.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).experts.(\\d+).(w3|up_proj).",
                       "layers.\\1.ff.experts.\\3.w_gated.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).experts.(\\d+).(w2|down_proj).",
                       "layers.\\1.ff.experts.\\3.w_out.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).gate.weight",
                       "layers.\\1.ff.router.weight", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).shared_expert.(w1|gate_proj).",
                       "layers.\\1.ff.shared_expert.w_in.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).shared_expert.(w3|up_proj).",
                       "layers.\\1.ff.shared_expert.w_gated.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).shared_expert.(w2|down_proj).",
                       "layers.\\1.ff.shared_expert.w_out.", s)
            s = re.sub("model.layers.(\\d+).(mlp|block_sparse_moe).shared_expert_gate.",
                       "layers.\\1.ff.shared_expert_gate.", s)
        # CPMD MOE
        if "ffn.ffn.experts." in s:
            s = re.sub(
                "encoder.layers.(\\d+).ffn.ffn.experts.(\\d+).w_in.w_0.(.*)weight",
                "layers.\\1.ff.experts.\\2.w_in.\\3weight",
                s,
            )
            s = re.sub(
                "encoder.layers.(\\d+).ffn.ffn.experts.(\\d+).w_in.w_1.(.*)weight",
                "layers.\\1.ff.experts.\\2.w_gated.\\3weight",
                s,
            )
            s = re.sub(
                "encoder.layers.(\\d+).ffn.ffn.experts.(\\d+).w_out.(.*)weight",
                "layers.\\1.ff.experts.\\2.w_out.\\3weight",
                s,
            )
        s = re.sub("encoder.layers.(\\d+).ffn.ffn.router.weight",
                   "layers.\\1.ff.router.weight", s)

        # Deep seek
        s = re.sub("model.layers.([0-9]+).self_attn.", "layers.\\1.attn.", s)

        return "llama." + s
