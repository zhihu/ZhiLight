# coding=utf-8

import os
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
        parallel=False,
    )
    # model.load_model_pt(model_path)
    model.load_model_safetensors(model_path)
    print(f">>>Load model '{model_path}' finished in {time.time() - t0:.2f} seconds<<<")

    arg = GeneratorArg(
        beam_size=1,
        max_length=100,
        repetition_penalty=1.0,
    )
    with DynamicBatchGenerator(DynamicBatchConfig(), model) as generator:
        text = "<s>[INST] What is your favourite condiment? [/INST]"
        req_result = generator.generate(text, arg)
        print(req_result)


if __name__ == "__main__":
    main("/dev/shm/Mistral-7B-Instruct-v0.2")
