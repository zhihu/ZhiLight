# coding=utf-8
# Author: spetrel@gmail.com

import argparse

from zhilight.dynamic_batch import GeneratorArg, DynamicBatchConfig


def define_parser():
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--auto_model", type=int, default=0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quant", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=True)

    group = parser.add_argument_group("DynamicBatchConfig")
    group.add_argument("--max_batch", type=int, default=32)

    group = parser.add_argument_group("GeneratorArg")
    group.add_argument("--beam_size", type=int, default=1)
    group.add_argument("--max_length", type=int, default=1024)
    group.add_argument("--repetition_penalty", type=float, default=1.00)
    group.add_argument("--ngram_penalty", type=float, default=1.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--top_p", type=float, default=1.0)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--seed", type=int, default=0)

    parser.add_argument("--round", type=int, default=1)

    return parser


def generator_config_from_cmd(args):
    return DynamicBatchConfig(
        max_batch=args.max_length
    )


def generator_arg_from_cmd(args):
    return GeneratorArg(
        beam_size=args.beam_size,
        max_length=args.max_length,
        repetition_penalty=args.repetition_penalty,
        ngram_penalty=args.ngram_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed,
    )

