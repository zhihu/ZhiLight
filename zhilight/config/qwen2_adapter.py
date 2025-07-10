# coding=utf-8

from .dev_config import *
from .config_util import *


class Qwen2Adapter:
    @staticmethod
    def adapt(config: dict):
        if config["num_hidden_layers"] in [48, 28] and config["hidden_size"] in [5120, 3584]:
            m_size = '14b' if config["num_hidden_layers"] == 48 else '7b'
            print(f"##### Adapt qwen2 {m_size} config ########")
            set_envs({
                CHUNKED_PREFILL: 1,
                CHUNKED_PREFILL_SIZE: 512,
                FUSE_QKV: 1,
                FUSE_FF_IN: 1,
            })
            if os.environ.get(CHUNKED_PREFILL, "") == "1":
                set_envs({
                    HOST_REDUCE: 1,
                    HOST_REDUCE_COPY_ONLY: 1,
                    DUAL_STREAM: 1,
                    DUAL_STREAM_THRESHOLD: 100,
                })

        if config["num_hidden_layers"] == 80 and config["hidden_size"] == 8192:
            m_size = '72b' if config["intermediate_size"] == 29696 else '110b'
            print(f"##### Adapt qwen2 {m_size} config ########")
            set_envs({
                HIGH_PRECISION: 0,
                FUSE_QKV: 1,
                FUSE_FF_IN: 2,
                DUAL_STREAM: 1,
                REDUCE_TP_INT8_THRES: 1000,
            })
            if m_size == '110b':
                set_env(PRE_ALLOC_ALL_TOKEN, 0, "for 110b to reduce memory usage.")
            if get_quant_method(config) == "awq":
                set_env("AWQ_USE_EXLLAMA", 1)

        if config.get("rope_scaling", None):
            if "factor" in config["rope_scaling"] and config["rope_scaling"]["factor"] > 1.:
                if os.environ.get("DISABLE_ROPE_SCALING", "") == "1":
                    config.pop("rope_scaling")
                set_env(CHUNKED_PREFILL, 1, "for LONG context to reduce memory usage!")
                set_envs({CHUNKED_PREFILL_SIZE: 8192})
