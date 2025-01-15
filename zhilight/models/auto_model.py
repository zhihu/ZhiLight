# coding=utf-8
# Author: spetrel@gmail.com

import glob
import importlib
from zhilight.loader import LLaMALoader
from zhilight.llama import LLaMA

_CUSTOMIZED_MODELS = {
    "deepseek_vl_v2": "DeepseekVL2",
}

_TEXT_MODEL_TYPES = [
    "cohere",
    "deepseek_v2",
    "llama",
    "minicpm3",
    "qwen2",
]


def is_customized_model(model_type: str) -> bool:
    return model_type in _CUSTOMIZED_MODELS


class AutoModel:
    @classmethod
    def from_pretrained(cls, model_path):
        config: dict = LLaMALoader.load_llama_config(model_path)
        model_type = config.get("model_type", "")
        if model_type in _CUSTOMIZED_MODELS:
            mod = importlib.import_module(f"zhilight.models.{model_type}")
            customized_model_cls = getattr(mod, _CUSTOMIZED_MODELS[model_type])
            model: LLaMA = customized_model_cls(model_path, model_config=config, parallel=True)
        else:
            if model_type not in _TEXT_MODEL_TYPES:
                print(f"WARING: Loading unknown model type: {model_type}")
            model = LLaMA(model_path, model_config=config, parallel=True)

        if glob.glob(f"{model_path}/*.safetensors"):
            model.load_model_safetensors(model_path)
        else:
            model.load_model_pt(model_path)

        return model
