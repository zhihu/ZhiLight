# coding=utf-8

import os
import sys
import time

from zhilight import LLaMA, QuantConfig, QuantType
from zhilight.loader import LLaMALoader
from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator


def main(model_path):
    t0 = time.time()
    model_config = LLaMALoader.load_llama_config(model_path)
    model = LLaMA(
        model_path,
        model_config=model_config,
        quant_config=None,
        parallel=True,
    )
    # model.load_model_pt(model_path)
    model.load_model_safetensors(model_path)
    print(f">>>Load model '{model_path}' finished in {time.time() - t0:.2f} seconds<<<")

    arg = GeneratorArg(
        beam_size=1,
        max_length=100,
        repetition_penalty=1.0
    )
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    batch_config = DynamicBatchConfig(flash_attention=True)
    with DynamicBatchGenerator(batch_config, model) as generator:
        req_result = generator.generate(messages, arg)
        print(req_result)


if __name__ == "__main__":
    main("llama_3_8b_instruct_awq")
