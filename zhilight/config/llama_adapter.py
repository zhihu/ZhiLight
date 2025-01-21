# coding=utf-8

from .config_util import *
from .dev_config import *


class LLaMAAdapter:
    @staticmethod
    def adapt(config: dict):
        if config["rope_scaling"].get("rope_type", None) == "llama3":
            set_env(ROPE_CACHE, 1)
