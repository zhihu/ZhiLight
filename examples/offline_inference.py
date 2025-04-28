#
# Simple example of zhilight loading hugginface model for offline inference, more cases can be found in tests directory.
#
from zhilight import LLaMA
from zhilight.loader import LLaMALoader
from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator

model_path = "./Qwen2.5-72B-Instruct-GPTQ-Int4"

model_config = LLaMALoader.load_llama_config(model_path)

model = LLaMA(
    model_path = model_path,
    model_config = model_config,
    quant_config = None,
    parallel = True,
)

model.load_model_safetensors(model_path)

dyn_config = DynamicBatchConfig(
    max_batch=64
)

arg = GeneratorArg(
    max_length = 10,
    repetition_penalty = 1.0,
    ngram_penalty = 1.0,
    top_k = 0,
    top_p = 0.95,
    temperature = 1.0,
    seed = 42,
)

prompts = [
    "三星堆文明是外星文明吗？",
    "奥陌陌是不是外星飞船？"
]

with DynamicBatchGenerator(dyn_config, model) as generator:
    results = generator.batch_generate(prompts, arg)
    for i in range(len(prompts)):
        print(f"{prompts[i]} => {results[i].outputs[0].text}")