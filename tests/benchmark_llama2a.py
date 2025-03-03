# -*- coding: UTF-8 -*-

import argparse
import concurrent.futures
import json
import numpy as np
import os
import pandas as pd
import sys
import time
from threading import Lock
from typing import List, Optional, Tuple

from zhilight import LLaMA, QuantConfig, QuantType
from zhilight.loader import LLaMALoader
from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator, RequestResult


def auto_set_gpu(quant, num_gpu):
    if num_gpu == 8 and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,4,6,1,3,5,7'
        if quant:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
            os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,5,7'
        if quant >= 3:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
        print(f'Set devices to {os.environ["CUDA_VISIBLE_DEVICES"]}')


def load_model(model_path, quant=None, parallel=True, use_shm_cache=False):
    t0 = time.time()
    model_config = LLaMALoader.load_llama_config(model_path)
    if quant is None:
        quant = 3 if 'int4' in model_path else 2 if 'int8' in model_path else 0
    quant_config = QuantConfig(type=QuantType(quant))
    print(f"parallel={parallel} quant={quant}")

    model = LLaMA(
        f"{model_path}",
        model_config=model_config,
        quant_config=quant_config,
        parallel=parallel,
    )

    model.load_model_pt(f"{model_path}")

    print(f">>>Load model '{model_path}' finished in {time.time() - t0:.2f} seconds<<<")
    return model


def print_all(*a):
    print(*a, file=sys.stdout)
    # print(*a, file=sys.stderr)

messages = [{'role': 'system', 'content': 'You are a large language AI assistant built by Zhihu AI. 当前日期是: 2024年02月26日 星期一 14:10:21\n'}, {'role': 'user', 'content': "You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number.\n\nYour answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. \n\nHere are the set of contexts:\n\n1. “我认识你，早在你认识我之前。”“这话是什么意思？”地上的男人捂着伤口，一边蹭着向后退，一边睁大眼睛惊恐的看着我。我唇角勾起一抹微笑，一刀抹了男人的脖子。死人哪里需要知道那么多答案。——我是组织里的流放者，是实验中的“失败品”。被流放后，我被一个好心的奶奶收养，成为了f大的学生，后来奶奶也去世了。我就又成为了孤身一人。但是最近，我发现我被盯上了。后面的人自以为隐藏的很好，在人群中跟随着我的脚步。我察觉不对，随即绕进巷子，拐到一个静默的角落站定，听着他们的脚步从我身边逐渐远去，才慢悠悠的走出巷子回家。本来这件事对我现在的生活就像打水漂，经不起什么波澜。因为被流放的人都是被组织认定的实验失败品，大部分都活不过三个月。所以如果是那个组织的人，大抵不会因为我曾经“试验品”身份来抓捕我，于是我还是有恃无恐的每天正常上下学。直到我被教授委托去看一看最近“闹失踪”师兄，我才发觉，这件事大条了。我正在f大攻读生物化学博士，不务正业的研究人类的进化和脑神经的关系。也发表了几篇对国际生物学没什么影响的水论。虽然我没什么拿的出手的成绩，但是我的导师和师兄都算的上生物化学领域有名的人物。师兄家的门不正常的半掩着，我径直推门走进去。平日干净整洁的实验人家，现在像是遭了贼，凌乱不堪，甚至没有完整的落脚地。\n\n2. 1“我曾经见过你的。”听到这句话，我猝不及防回头，一个明媚的少女冲我挥了挥手，漏出标准的八颗牙齿。“小姐姐，就在那个时间通道里。”她小跑过来，兴奋着手舞足蹈。“我终于见到你啦！”我有些无语。心里想，这是哪里来的二次元少女？新式营销？地铁门开了，懒得理睬她直接走了进去找个位置坐下。她并没有被我的冷漠吓退，跟了进来坐在我旁边。“小姐姐你不记得我了吗？”她带着婴儿肥的脸蛋上有些失望。“就在那里，我看见你是第一个进入时间通道的人！”她声音甜甜的，细嫩而白净的手比划着。“你认错人了。”我回答。“我平常不打游戏。”她垂头丧气，“不是游戏啊......”但很快又打起精神，“好吧！那我们重新认识一下。”我看向她，她圆滚滚的眼睛里满是认真，伸出手：“小姐姐你好，我叫言俒俒。”奇怪的名字。地铁上的目光都被吸引过来，为了不引起太多注意，我回握她，“你好，言俒俒，我叫刘岐。”“刘岐？哇，好帅气的名字。”言俒俒赞叹。未完，随缘更新\n\n3. 我与他年少相识，一眼便定了终生后来我家道中落， 我家几百口人，都死在了那一天，只有我被他救了下来，我被他养在他的东宫里，只有他和他的母妃知道我的身份，他的母妃和我娘是闺中密友，所以得知我的身份后，便让他把我养在了东宫。及笄那天，我有了另一个身份，我成了镇国大将军的养女，嫁给了他，当了他的侧妃后来他处心积虑地登上了皇位，因东宫只有我一个妃子， 所以我一入后宫便是贵妃，封号安乐，那些大臣们纷纷上奏让自己的女儿进入后宫，他却一人没收，顶着压力 让这后宫只有我一人后来我有了身孕，他笑得像孩子般，当皇帝后，除了和我呆在一起，他都是阴沉沉的，带着天子不怒自威的气势，可跟我呆在一块，他总是满面春风，因为我怀了皇嗣，我变成了皇贵妃，他天天摸着我胎中还未出生的孩子，同他说着话，十月怀胎我将孩子生了下来，生产那天，他呆在产房外，不停的走动，简直比我还着急，孩子生下来了，是个健康的男孩，他一出生便被封了太子，成了未来做皇帝的人，看着我以为刚生产完虚弱的脸他哭着对我说，音儿，我们不生了，我不要小公主了，我只要你，我只要你后来他凭借他的势力，将当年那件，涂了我家满门的事，又查了好几遍，证明了我爹不是贪官，他是清清白白的，他是个好官，他从来没有和外国交易，于是我又恢复了丞相嫡女的身份，我成了皇后。\n\n4. 注：. 这些是用R语言爬出来的。. 代码见 Penroseasdf：☕R语言 | 用jsonlite包爬知乎AJAX动态加载网页. 部分链接问题相同但是点进去回答是不同的哦. 10月17日更新，以后会按照文本内容来归类，方便查找 （大概） 。. “他想修仙，可昆仑太远…”为开头能写出怎样的 ...\n\n5. 第一种开头方法：名言警句式开头 这是用得非常广的一种开头，就是在作文开篇引用名言警句，从而让作文的开头显得权威、深刻或者有文采，从而夺人眼球，起到开篇不凡的效果。 适用范围：命题作文，话题作文，材料作文，新材料（任务驱动型）作文 【高考模拟题】 四年前，27岁的海林突然接到了家中电话，父亲需要30万元救急。 当晚，海林在自己的微信公众号上写了一篇名为《此后数月经年我做一个感恩的人》的文章：我需要30万，我寻找300位朋友，每个人借我1000元。 只接受微信转账……我会在以后的某一天还回去。 ”第二天 海林就收满了30万元的自助捐款，捐款者包括认识的朋友，也包括素昧平生的网友。 此后几年，海林靠着自己的收入，一点点还清了这30万元。 这件事被媒体报道后引发热议。 对此，你是怎么看的？\n\n6. 写作技巧|满分作文的7种开头妙法（附：技法、范例）. 俗话说：“万事开头难！. ”写作文也是，开头是在给文章造气氛、定调子，要给读者留下深刻的第一印象，因而十分重要。. 作文开头如果能恰到好处，常常能一下子抓住读者，也能增加文章的亮点。. 所以 ...\n\n7. 对于如何吸引眼球的开头，这确实是一个涉及许多因素和技巧的问题。以下是我对这个问题的一些思考和建议。首先，我们需要明白一个道理，那就是开头并不是文章的全部。尽管一个好的开头可以吸引读者的眼球，但如果文章的内容无法保持这种吸引力，那么\n\nRemember, don't blindly repeat the contexts verbatim, and you must write in Chinese language. Here is the user question:\n以你是第 1 个我认识的人为开头，写一篇不少于 1000 字的文章"}]

def simulate_qps_generate(generator: DynamicBatchGenerator, s_qps, input_texts, arg):
    print(f"Simulate request speed {s_qps} req/s")
    lock = Lock()
    req_results: List[RequestResult] = [None] * len(input_texts)
    time_elapses = []
    begin_ts = time.time()

    def gen(i, txt):
        req_res = generator.generate(txt, arg=arg)
        with lock:
            req_results[i] = req_res
            time_elapses.append(req_res.outputs[0].time_elapsed)
            finished = len(time_elapses)
        qps = finished / (time.time() - begin_ts)
        p50 = np.percentile(np.array(time_elapses), 50)
        p90 = np.percentile(np.array(time_elapses), 90)
        p95 = np.percentile(np.array(time_elapses), 95)
        p99 = np.percentile(np.array(time_elapses), 99)
        print(f"\rFinish {finished}/{len(input_texts)}, QPS={qps:.3f}, "
              f"P50={p50:.1f}, P90={p90:.1f}, P95={p95:.1f}, P99={p99:.1f}", end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        for idx, text in enumerate(input_texts):
            executor.submit(gen, idx, text)
            time.sleep(1. / s_qps)
        executor.shutdown(wait=True)
    print("\nDone!")
    return req_results


def run(args):
    model: LLaMA = load_model(args.model_path, quant=args.quant, parallel=args.parallel, use_shm_cache=args.shm_cache)
    print("############## begin run ##############")
    ts0 = time.perf_counter()

    dyn_config = DynamicBatchConfig(
        max_batch=args.max_batch,
        max_beam_size=args.beam_size,
        task_queue_size=20,
        rag_buffer=True,
        ignore_eos=bool(args.length_name),
        high_precision=-1,
        reserved_work_mem_mb=1250
    )
    arg = GeneratorArg(
        beam_size=args.beam_size,
        max_length=args.max_length,
        repetition_penalty=args.repetition_penalty,
        ngram_penalty=args.ngram_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed,
    )
    if args.in_file:
        df = pd.read_excel(args.in_file)
        df = df[:args.num_prompts]
        input_texts_d = [txt for txt in df[args.prompt_name]]
        input_texts = input_texts_d * args.round
        if args.round > 1:
            df[args.prompt_name] = input_texts
    else:
        df = pd.DataFrame()
        input_texts = [messages] * args.round

    with DynamicBatchGenerator(dyn_config, model) as generator:
        if args.first_delay:
            arg.max_length = 1
            for _ in range(3):
                print(generator.generate(messages, arg))
            return
        if args.simulate_qps > 0:
            batch_res = simulate_qps_generate(generator, args.simulate_qps, input_texts, arg)
        else:
            input_tokens_num = [x for x in df["input_tokens_num"]] if args.length_name else None
            max_lengths = [x for x in df[args.length_name]] if args.length_name else None
            if max_lengths and args.top_p < 1.:
                raise ValueError("max_lengths should be used with beam search only.")
            batch_res = generator.batch_generate(
                input_texts,
                arg,
                max_in_lengths=input_tokens_num,
                max_out_lengths=max_lengths,
                prepend_input=False)
    batch_res_d = batch_res[:len(input_texts)]
    output_texts = [r.outputs[0].text for r in batch_res_d]
    input_tokens_num = [r.input_tokens_num for r in batch_res_d]
    output_tokens_num = [r.outputs[0].output_tokens_num for r in batch_res_d]

    elapsed_time = time.perf_counter() - ts0

    # save result
    df[args.result_name] = output_texts
    df["input_tokens_num"] = input_tokens_num
    df["output_tokens_num"] = output_tokens_num
    df.to_excel(args.out_file)

    # print summary
    total_input_token = sum(r.input_tokens_num for r in batch_res)
    total_out_tokens = sum(r.outputs[0].output_tokens_num for r in batch_res)
    total_num_tokens = total_input_token + total_out_tokens

    print_all(
        f"[ZhiLight] reqNum={len(input_texts)} Input {total_input_token}; Out {total_out_tokens}; Total {total_num_tokens} Time:{elapsed_time:.2f}s")
    print_all(f"Throughput: {len(input_texts) / elapsed_time:.3f} req/s, "
              f"Total {total_num_tokens / elapsed_time:.1f} tokens/s; Out {total_out_tokens / elapsed_time:.1f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--shm_cache", type=bool, required=False)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--prompt_name", type=str, required=True)
    parser.add_argument("--result_name", type=str, required=True)
    parser.add_argument("--length_name", type=str, default=None)
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to process.")
    parser.add_argument("--round", type=int, default=1, help="Number of round to process.")
    parser.add_argument("--print_num", type=int, default=0)
    parser.add_argument("--max_batch", type=int, default=128)
    parser.add_argument("--quant", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=True)

    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--ngram_penalty", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--simulate_qps", type=float, default=0)
    parser.add_argument("--first_delay", type=int, default=0)

    args = parser.parse_args()

    run(args)
