# -*- coding: UTF-8 -*-

import argparse
import json
import os
import sys
import time
import torch

from zhilight import LLaMA, QuantConfig, QuantType
from zhilight.load_tensor_util import load_llama_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='act_scales.pt',
                        help='where to save the act scales')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--jsonl', type=int, default=1)
    args = parser.parse_args()
    return args


def read_dataset(ds_path, step, jsonl=True):
    def join_text(data):
        return "".join(
            [
                data.get("input", "").strip(),
                data.get("output", "").strip(),
                data.get("title", "").strip(),
                data.get("question", "").strip(),
                data.get("answer", "").strip(),
                data.get("abstract", "").strip(),
                data.get("text", "").strip(),
                data.get("code", "").strip(),
            ]
        ).strip()

    with open(ds_path, "r", encoding="utf-8") as f:
        if jsonl:
            ds = [json.loads(s) for s in f.readlines() if not s.startswith('{"dataset_name":')]
        else:
            ds = json.load(f)
            if not isinstance(ds, list):
                raise ValueError("Not a list")

    print(f"Dataset contains {len(ds)} lines.")
    ds = [join_text(ds[i]) for i in range(0, len(ds), step)]
    ds = [s for s in ds if s]
    return ds


def main():
    args = parse_args()
    dataset = read_dataset(args.dataset_path, args.step,args.jsonl)

    model_path = args.model_path
    with open(f"{model_path}/config.json") as f:
        model_config = json.load(f)
    if "dim_ff" not in model_config:
        raise ValueError("Not a caterpillar config file")
    model_config["new_vocab"] = True

    model = LLaMA(
        f"{model_path}/vocabs.txt",
        model_config=model_config,
        parallel=True,
    )
    load_llama_cache(model, model_path)
    # model.load_model_pt(model_path)
    act_scales = model.calc_act_scales(dataset)
    act_scales2 = {name: torch.from_numpy(nd_arr).half() for name, nd_arr in act_scales.items()}
    torch.save(act_scales2, args.output_path)
    print(f"Done save scales to {args.output_path}")


if __name__ == '__main__':
    main()
