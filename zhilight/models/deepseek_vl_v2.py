# coding=utf-8
# Author: spetrel@gmail.com

import os
import pdb
import sys
import time
import torch
from torch import cuda

try:
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
except Exception as e:
    if "deepseek_vl2" in str(e):
        print(e)
        print("You can install it from https://github.com/deepseek-ai/DeepSeek-VL2")
        sys.exit(1)
    else:
        raise e

import PIL.Image
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Dict, Optional, Union
from zhilight.config.dev_config import *
from zhilight.config.deepseek_adapter import DeepseekV2Adapter
from zhilight.llama import LLaMA, LLaMAModelConfig, QuantConfig, DistConfig
from zhilight.loader import LLaMALoader
from zhilight.utils.image_utils import load_image

def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path_or_url_or_pil in message["images"]:
            pil_img = load_image(image_path_or_url_or_pil)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class ModelAdapter(torch.nn.Module):
    def __init__(self, c_model, dtype):
        super().__init__()
        self.c_model = c_model
        self.dtype = dtype

    """ interface of PreTrainedModel"""
    def get_input_embeddings(self):
        c_model = self.c_model
        dtype = self.dtype

        def func(ids):
            py_ids = ids.flatten().cpu().tolist()
            emb = c_model.get_input_embeddings(py_ids)
            return torch.from_numpy(emb).unsqueeze(0).to(device='cuda:0', dtype=dtype)
        return func


class DeepseekVL2(LLaMA):
    """
    deepseek_vl_v2 ZhiLight implementation.
    This multi-modal model contains two parts: vision model and language model.
      vision model: Re-use the original implementation from https://github.com/deepseek-ai/DeepSeek-VL2
      language model: Re-implement by ZhiLight to speed up.
    """
    def __init__(
            self,
            model_path: str,
            model_config: Optional[LLaMAModelConfig] = None,
            quant_config: Optional[QuantConfig] = None,
            parallel: Union[DistConfig, int, bool] = True,
            **kwargs
    ):
        if model_config is None:
            model_config = LLaMALoader.load_llama_config(model_path)
        self.dtype = model_config["torch_dtype"]
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        # # Load vision model directly by transformers
        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.vl_gpt.language = None
        for param in self.vl_gpt.parameters():
            param.requires_grad = False
        self.vl_gpt.cuda(0).eval()  # TODO: optimize for multi GPUs

        visual_model_size = sum(p.nbytes for p in self.vl_gpt.parameters())
        torch_allocated = cuda.memory_allocated(0)
        torch_allow = torch_allocated + (800 << 20)
        total_memory = cuda.get_device_properties(0).total_memory
        cuda.set_per_process_memory_fraction(torch_allow / total_memory)
        print(f"cuda.set_per_process_memory_fraction({torch_allow / total_memory})")
        set_env(RESERVE_MEM_MB, int(torch_allow / 1024000) + 1500)

        self._init_processor(model_path, None)

        # Init language model
        language_config: dict = self.vl_gpt.config.language_config.to_dict()
        DeepseekV2Adapter.adapt(language_config)
        super().__init__(
            model_path,
            model_config=language_config,
            quant_config=quant_config,
            parallel=parallel,
            tokenizer=self.hf_processor.tokenizer,
            device_id=-1,
            memory_limit=0)

        # REPLACE language model with our implementation, for get input embeddings
        self.vl_gpt.language = ModelAdapter(self._model, self.dtype)

    def load_model_safetensors(self, model_dir, pattern="*.safetensors"):
        state_dict = LLaMALoader.load_safetensors(model_dir, pattern)
        state_dict = {k[len('language.'):]: v for k, v in state_dict.items() if k.startswith('language.')}
        self.load_state_dict_pt(state_dict)

    def _init_processor(self, model_path, config):
        self.hf_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)

    def process_inputs(self, messages: List[dict]):
        t0 = time.time()
        with torch.no_grad():
            with torch.cuda.device(0):
                ret = self._process_inputs(messages)
        # print(f"process_inputs task {time.time() - t0:.3f}")
        return ret

    def _process_inputs(self, messages: List[dict]):
        assert isinstance(messages, list) and isinstance(messages[0], dict)
        pil_images = load_pil_images(messages)
        prepare_inputs = self.hf_processor.__call__(
            conversations=messages,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to("cuda:0", dtype=self.dtype)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        token_ids = prepare_inputs.input_ids.flatten().cpu().tolist()
        inputs_embeds = inputs_embeds.squeeze(0).to(torch.float).detach().cpu().numpy()
        # if len(token_ids) < 100:
        #     print("token_ids:", token_ids)
        #     print("inputs_embeds.shape:", inputs_embeds.shape)
        #     print("inputs_embeds:", inputs_embeds)
        #     inputs_embeds = None
        return (token_ids,
                None,
                inputs_embeds,
                None,
                )
