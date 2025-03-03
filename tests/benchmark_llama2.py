# -*- coding: UTF-8 -*-

import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional, Tuple

from zhilight import LLaMA, QuantConfig, QuantType
from zhilight.llama import LLaMAModelConfig

try:
    from transformers import LlamaTokenizer, LlamaForCausalLM
except:
    LlamaTokenizer = None
    LlamaForCausalLM = None
import numpy as np
import time
import torch


def load_model(ver="7b"):
    assert LlamaTokenizer, "llama tokenizer load failed, pip install transformer"

    t0 = time.time()
    model_dir = "/mnt/data/user/tc_agi/user/cgq/llama2-70b-chat"
    model_config_70b: LLaMAModelConfig = {
        "num_layers": 80,
        "dim_model": 8192,
        "num_heads": 64,
        "dim_head": 128,
        "dim_ff": 28672,
        "vocab_size": 32000,
        "eps": 1e-5,
        "num_kv_heads": 8,
    }

    model = LLaMA(
        model_dir,
        model_config=model_config_70b,
        quant_config=None,
        parallel=True,
    )

    model.load_model_pt(f"{model_dir}")

    print(f">>>model load finished in {time.time() - t0:.2f} seconds<<<")
    return model


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        cache_sampled=True,
) -> List[Tuple[str, int, int]]:
    cache_json_file = "sampled_req.json"
    if cache_sampled and os.path.isfile(cache_json_file):
        with open(cache_json_file) as f:
            sampled_requests = json.load(f)
        return sampled_requests[:num_requests]

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = [tokenizer.encode(p) for p in prompts]
    completions = [completion for _, completion in dataset]
    completion_token_ids = [tokenizer.encode(p) for p in completions]
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, 3000)
    if cache_sampled:
        print(f"Write sampled requests to cache file {cache_json_file}.")
        with open(cache_json_file, "w") as f:
            json.dump(sampled_requests, f, ensure_ascii=False)

    return sampled_requests[:num_requests]


def print_all(*a):
    # print(*a, file=sys.stdout)
    print(*a, file=sys.stderr)


def bench(args):
    model: LLaMA = load_model("70b-org")
    if not os.path.isfile(args.dataset):
        print(f"{args.dataset} is not file")
        print("You can download the ShareGPT dataset by running:")
        print("wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json")
        sys.exit(1)

    tokenizer = model._base._tokenizer
    sampled_requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.cache_sampled)

    texts = [x[0] for x in sampled_requests]
    max_lengths = [x[2] for x in sampled_requests]
    print("############## begin bench ##############")
    ts0 = time.perf_counter()
    from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator
    dyn_config = DynamicBatchConfig(max_batch=args.max_batch,
                                    max_beam_size=args.beam_size,
                                    task_queue_size=20,
                                    nccl=False,
                                    rag_buffer=True,
                                    ignore_eos=True)
    max_length = args.max_length if args.max_length > 0 else max_lengths
    res = model.dynamic_batch_inference(texts,
                                        dyn_config=dyn_config,
                                        max_length=max_length,
                                        beam_size=args.beam_size,
                                        fetch_tokens_level=1,
                                        prepend_input=False)
    elapsed_time = time.perf_counter() - ts0

    # print summary
    total_input_token = sum(r["input_tokens_num"] for r in res)
    total_out_tokens = sum(r["output_tokens_num"] for r in res)
    total_num_tokens = total_input_token + total_out_tokens
    print_all(
        f"[ZhiLight] reqNum={args.num_prompts} Input {total_input_token}; Out {total_out_tokens}; Total {total_num_tokens} Time:{elapsed_time:.2f}s")
    print_all(f"Throughput: {len(texts) / elapsed_time:.3f} req/s, "
              f"Total {total_num_tokens / elapsed_time:.1f} tokens/s; Out {total_out_tokens / elapsed_time:.1f} tokens/s")

    # print result
    for i in range(args.print_num):
        print_all(res[i])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    # You can download the ShareGPT dataset by running:
    # wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num_prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--print_num", type=int, default=0)
    parser.add_argument("--max_batch", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--use_shm_cache", action="store_true")
    parser.add_argument("--cache_sampled", action="store_true")

    args = parser.parse_args()

    bench(args)
