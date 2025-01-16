import copy
import gc
import json
import logging
import math
import os
import sys
import time
from queue import Queue
from typing import List, Optional, Union, Tuple
from threading import Thread
from . import C
from .quant import QuantConfig, QuantType


class DynamicBatchConfig:
    def __init__(
            self,
            max_batch=20,
            max_beam_size=4,
            task_queue_size=8,
            max_total_token=4096,
            seed=0, # deprecated
            eos_id=1,
            bos_id=0,
            unk_id=0,
            first_batch=1,
            sort_by_len=0,
            nccl=-1,  # deprecated, always True
            rag_buffer=True,  # deprecated, always True
            ignore_eos=False,
            keep_eos=False,
            reserved_work_mem_mb=int(os.environ.get("reserved_work_mem_mb", 1250)),
            high_precision=1,
            flash_attention=True,
            enable_prompt_caching=False,
    ):
        self.max_batch = max_batch
        self.max_beam_size = max_beam_size
        self.task_queue_size = max(task_queue_size, first_batch)
        self.max_total_token = max_total_token
        self.seed = seed
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.first_batch = first_batch
        self.sort_by_len = int(sort_by_len)
        self.nccl = int(nccl)
        self.rag_buffer = bool(rag_buffer)
        self.ignore_eos = bool(ignore_eos)
        self.keep_eos = keep_eos
        self.reserved_work_mem_mb = reserved_work_mem_mb
        self.high_precision = high_precision
        self.flash_attention = flash_attention
        self.enable_prompt_caching = enable_prompt_caching or "1" == os.environ.get("enable_prompt_caching", "")
        if enable_prompt_caching and not rag_buffer:
            raise ValueError("prompt_caching must use with rag buffer")

    def c_config(self):
        config = C.DynBatchConfig()
        config.max_batch = self.max_batch
        config.max_beam_size = self.max_beam_size
        config.task_queue_size = self.task_queue_size
        config.max_total_token = self.max_total_token
        config.seed = self.seed
        config.unk_id = self.unk_id
        config.bos_id = self.bos_id
        config.eos_id = self.eos_id
        config.first_batch = self.first_batch
        config.nccl = self.nccl
        config.rag_buffer = self.rag_buffer
        config.ignore_eos = self.ignore_eos
        config.keep_eos = self.keep_eos
        config.reserved_work_mem_mb = self.reserved_work_mem_mb
        config.high_precision = self.high_precision
        config.flash_attention = self.flash_attention
        config.enable_prompt_caching = self.enable_prompt_caching
        return config

    def __repr__(self) -> str:
        return "DynamicBatchConfig(max_batch=%d, max_beam_size=%d, task_queue_size=%d, " \
               "max_total_token=%d, seed=%d, bos_id=%s, eos_id=%s, nccl=%s, rag_buffer=%s)" % \
               (self.max_batch, self.max_beam_size, self.task_queue_size, self.max_total_token,
                self.seed, self.bos_id, self.eos_id, self.nccl, self.rag_buffer)


class GeneratorArg:
    def __init__(
            self,
            beam_size: int = 1,
            max_length: int = 100,
            repetition_penalty: float = 1.0,
            ngram_penalty: float = 1.0,
            seed: int = 0,
            temperature: float = 1.0,
            num_results: int = 1,
            top_p: float = 1.0,  # use random sample if top_p < 1. or top_k > 0
            top_k: int = 0,
            bee_answer_multi_span: Optional[bool] = None,
            presence_penalty: float = 0.,
            top_logprobs: int = 0,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.presence_penalty = presence_penalty
        self.repetition_penalty = 1. if presence_penalty else repetition_penalty
        self.ngram_penalty = 1. if presence_penalty else ngram_penalty
        self.seed = seed
        self.temperature = float(temperature)
        self.num_results = int(num_results)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.bee_answer_multi_span = bee_answer_multi_span
        self.top_logprobs = top_logprobs

        if self.is_random:
            self.seed = seed or 42
        elif not self.is_random and self.seed:
            logging.warning("BeamSearch ignore seed")
            self.seed = 0

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def is_random(self):
        return self.top_p < 1. or self.top_k > 0

    def with_max_length(self, new_len):
        obj = self.copy()
        obj.max_length = new_len
        return obj


class StreamResultType:
    Incremental = 1
    AllCurrent = 2  # full update
    Final = 3


class GenerativeOutput:
    def __init__(self, token_ids, score, time_elapsed, first_token_delay=0, top_logprobs=None):
        self.token_ids: List[int] = token_ids
        self.score: float = score
        self.time_elapsed: float = time_elapsed / 1000
        self.first_token_delay: float = first_token_delay / 1000
        self.top_logprobs = top_logprobs
        self.text = ''

    def decode(self, tokenizer, prefix_input=None):
        self.text = tokenizer.decode(self.token_ids)
        if bool(prefix_input):
            self.text = prefix_input + self.text
        return self

    @property
    def output_tokens_num(self):
        return len(self.token_ids)

    def print_top_logprobs(self, tokenizer=None, start=0, end=-1, highlight_prob=1e-100, amb_only=False):
        top_logprobs = self.top_logprobs[start:end]
        highlight_prob = math.log(highlight_prob)
        for i, logprob_dict in enumerate(top_logprobs):
            logprob = sorted(logprob_dict.items(), key=lambda x: x[1], reverse=True)
            if logprob[0][1] < highlight_prob:
                print("\033[92m", end='')
            elif amb_only:
                continue
            print(f"[T{start+i:<2}] ", end='')
            for k, v in logprob:
                token = tokenizer.decode([k]).replace('\n', '\\n') if tokenizer else ''
                print(f"{k:<5}:\t'{token}',  {v:.5f};  \t", end='')
            print("\033[00m")

    def __repr__(self) -> str:
        out_tps = len(self.token_ids) / max(0.001, self.time_elapsed - self.first_token_delay)
        return (f"GenerativeOutput(text={repr(self.text)}, "
                f"first_token_delay={self.first_token_delay}, "
                f"out_tps={out_tps:.3f}, "
                f"score={self.score:.6f}, "
                f"output_tokens_num={len(self.token_ids)}, "
                f"time_elapsed={self.time_elapsed})")


def _convert_output(t) -> GenerativeOutput:
    """
     @:param t from c++ implementation: py_batch_generator.cpp convert_output_to_py()
    """
    if isinstance(t, tuple):
        # LLaMA result
        return GenerativeOutput(t[0], t[1], t[2], t[3], t[4])
    else:
        raise RuntimeError("Unexpected type")


class RequestResult:
    def __init__(self, prompt, outputs, input_tokens_num):
        self.prompt = prompt
        self.outputs: List[GenerativeOutput] = outputs
        self.input_tokens_num = input_tokens_num

    @staticmethod
    def from_cpp_result(prompt, c_outputs, input_tokens_num):
        """
         @:param t from c++ implementation: py_batch_generator.cpp convert_outputs_to_py()
        """
        assert isinstance(c_outputs, list), "c_outputs should be a list"
        outputs = [_convert_output(x) for x in c_outputs]
        return RequestResult(prompt, outputs, input_tokens_num)

    @staticmethod
    def from_cpp_stream_result(prompt, t, input_tokens_num):
        """
         @:param t from c++ implementation: py_batch_generator.cpp get_result()
        """
        assert isinstance(t, tuple), "cpp stream result should be a tuple"
        update_flag, _, _, final_results = t
        assert update_flag == StreamResultType.Final, "not a final result"
        assert isinstance(final_results, list), "final_results should be a list"
        outputs = [_convert_output(x) for x in final_results]
        return RequestResult(prompt, outputs, input_tokens_num)

    def __repr__(self) -> str:
        return f"RequestResult(outputs={self.outputs}, input_tokens_num={self.input_tokens_num})"


class StreamResult:
    def __init__(self, update_flag, tokens, score, final_results):
        self.update_flag: StreamResultType = update_flag
        self.tokens: Union[List[int], str] = tokens  # LLaMA: list[int]; CpmBee: str
        self.score: float = score
        self.final_results: List[GenerativeOutput] = final_results

    @staticmethod
    def from_cpp(t):
        """
         @:param t from c++ implementation: py_batch_generator.cpp get_result()
        """
        assert isinstance(t, tuple), "cpp stream result should be a tuple"
        assert isinstance(t[3], list), "final_results should be a list"
        final_results = [_convert_output(x) for x in t[3]]
        return StreamResult(t[0], t[1], t[2], final_results)


class StreamHandler:
    def __init__(self, task, arg, input_tokens, tokenizer):
        self.task = task
        self.arg = arg
        self.input_tokens = input_tokens
        self.task_id = id(self.task)
        self._tokenizer = tokenizer
        self.token_cache: Union[List[int], str] = []  # LLaMA: list[int]; CpmBee: str
        self.decoded_len = 0
        self.ts0 = time.time()

    def _decode(self, ids, increasing=True):
        if isinstance(ids, tuple):
            ids = ids[0]
        if len(ids) == 1 and increasing and hasattr(self._tokenizer, "convert_ids_to_tokens"):
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            sentence_piece_underline = "▁"
            if sentence_piece_underline in tokens[0]:
                # keep space for english
                return tokens[0].replace(sentence_piece_underline, " ")
        if self._tokenizer is not None:
            return self._tokenizer.decode(ids)
        return ids

    def has_result(self):
        return self.task.has_result()

    def get_result(self, timeout: float = 0.) -> StreamResult:
        return StreamResult.from_cpp(self.task.get_result(timeout))

    def cancel(self):
        self.task.cancel()

    @property
    def input_tokens_num(self):
        return self.task.input_tokens_num()

    @property
    def output_tokens_nums(self) -> List[int]:
        return self.task.output_tokens_nums() or [len(self.token_cache)]

    def decode_stream_res(self, res: StreamResult, increasing=True):
        res_id = self.task_id
        time_elapsed = time.time() - self.ts0
        flag, tokens, score = res.update_flag, res.tokens, res.score
        if flag == StreamResultType.Final:
            # TODO: handle multiple results
            output = res.final_results[0].decode(self._tokenizer).text
            return StreamResultType.Final, res_id, output, score, time_elapsed

        if flag == StreamResultType.Incremental:
            self.token_cache += tokens  # append
        else:
            assert flag == StreamResultType.AllCurrent
            self.token_cache = tokens  # replace

        if flag == StreamResultType.Incremental and increasing:
            char = self._decode(self.token_cache[self.decoded_len:])
            if char.startswith("�") or char.startswith(" �"):
                if len(char) >= 3:
                    logging.error("StreamDecode bug")
                    char = char.replace("�", "")
                else:
                    char = ''
            elif char:
                self.decoded_len = len(self.token_cache)
            return flag, res_id, char, score, time_elapsed
        else:
            output = self._decode(self.token_cache, increasing=False)
            self.decoded_len = len(self.token_cache)
            # take out last 1~2 special token
            if len(self.token_cache) >= 2 and not self._decode(self.token_cache[-2:]):
                self.decoded_len = len(self.token_cache) - 2
            elif not self._decode(self.token_cache[-1:]):
                self.decoded_len = len(self.token_cache) - 1
            while output.endswith("�"):
                output = output[:len(output) - 1]
                self.decoded_len -= 1
            return StreamResultType.AllCurrent, res_id, output, score, time_elapsed


class PrintStreamHandler(StreamHandler):
    def handle(self, input_text):
        while True:
            flag, res_id, output, score, time_elapsed = self.decode_stream_res(self.get_result())
            output = output.replace("\n", "\\n")
            if flag == StreamResultType.Incremental:
                if output:
                    print(output, flush=True, end="")
                continue
            print(f"\r{input_text}{output}", flush=True, end="")
            if flag == StreamResultType.Final:
                print()
                return


class DynamicBatchGenerator:
    def __init__(self, config: DynamicBatchConfig, model):
        self.config = config
        self.model = model
        self.c_model = model._model
        self._tokenizer = model._tokenizer if hasattr(model, "_tokenizer") else None
        self.config.bos_id = self._tokenizer.bos_token_id
        self.config.eos_id = self._tokenizer.eos_token_id
        if hasattr(model, "tokenizer_config"):
            tokenizer_cfg: dict = model.tokenizer_config
            if "chat_template" in tokenizer_cfg and "eos_token" in tokenizer_cfg and \
                    tokenizer_cfg["eos_token"] != "</s>" and \
                    "added_tokens_decoder" in tokenizer_cfg:
                chat_eos_token = tokenizer_cfg["eos_token"]
                for token_id, added_tokens in tokenizer_cfg["added_tokens_decoder"].items():
                    if chat_eos_token == added_tokens["content"]:
                        self.config.eos_id = int(token_id)
                        print(f"Set chat model eos_id to {token_id}")

        model_config = model._config if hasattr(model, "_config") and isinstance(model._config, dict) else {}
        max_token = model_config.get("max_token", 4096)

        is_int4 = hasattr(model, "_quant_config") and model._quant_config and \
                  model._quant_config.get("type", None) == QuantType.Int4
        if is_int4 and "max_token" in model_config and max_token < self.config.max_total_token:
            logging.warning("self.config.max_total_token bigger than model._config['max_token']")
            self.config.max_total_token = max_token
        # print(f"max_position_embeddings: {max_position_embeddings}")
        # self.config.max_total_token = max(self.config.max_total_token, max_position_embeddings)

        self._c_generator = C.BatchGenerator(self.config.c_config(), self.c_model)
        self._thread = None
        self.print_queue_threshold = int(os.environ.get("print_queue_threshold", 0))
        self._do_verify = int(os.environ.get("VERIFY_MAX_TOKEN", 1)) > 0
        if self._do_verify:
            self.start()
            arg = GeneratorArg(max_length=1)
            task = self.to_c_task([888] * (config.max_total_token - 1), arg)
            self.generate_c(task, arg)
            self._do_verify = False
            print("Done Verify max_token")

    @staticmethod
    def to_c_task(input_tokens: List[int], arg: GeneratorArg, stream=0) -> C.SearchTask:
        return C.SearchTask(
            input_tokens,
            arg.beam_size,
            arg.max_length,
            arg.presence_penalty,
            arg.repetition_penalty,
            arg.ngram_penalty,
            arg.seed is not None and arg.seed != 0,
            arg.seed or 0,
            arg.temperature,
            arg.num_results,
            arg.top_p,
            arg.top_k,
            bool(arg.bee_answer_multi_span),
            arg.top_logprobs,
            int(stream))

    def check_queue_busy(self):
        queue_size = self._c_generator.queue_size()
        if queue_size > self.print_queue_threshold:
            logging.warning(f"High pressure: active_size={self._c_generator.active_size()} queue_size={queue_size}")

    def _encode(self, data: Union[str, List[dict]]) -> List[int]:
        if (
            isinstance(data, list)
            and isinstance(data[0], dict)
            and hasattr(self._tokenizer, "apply_chat_template")
        ):
            # https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages
            prompt = self._tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)
            # print("prompt", prompt)
            input_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
            # print("input_tokens", input_tokens)
        else:
            input_tokens = self._tokenizer.encode(data)
        return input_tokens

    def _process_inputs(self, data: Union[str, dict, List[dict]], arg: GeneratorArg, stream=0):
        t = self.model.process_inputs(data)
        if t is not None:
            # multi-modal model, feed extra fields
            ids, position_ids, embeddings, pos_delta = t
            c_task = self.to_c_task(ids, arg, stream=stream)
            if position_ids is not None:
                c_task.set_position_ids(position_ids)
            if embeddings is not None:
                assert 0 == int(os.environ.get("CHUNKED_PREFILL", 0)), "Feed embeddings with chunking is not supported."
                c_task.set_input_embeddings(embeddings)
            if pos_delta is not None:
                c_task.set_position_delta(pos_delta)
            return c_task, ids
        else:
            input_tokens = self._encode(data)
            return self.to_c_task(input_tokens, arg), input_tokens

    def generate(
        self,
        data: Union[str, dict, List[dict]],
        arg: GeneratorArg = GeneratorArg(),
        block: bool = True,
        prepend_input: bool = False,
        timeout: float = 0,
    ):
        c_task, _ = self._process_inputs(data, arg)

        req_result = self.generate_c(c_task, arg, block, timeout=timeout)

        for output in req_result.outputs:
            output.decode(self._tokenizer, data if prepend_input else None)

        return req_result

    def generate_c(
        self,
        c_task: C.SearchTask,
        arg: GeneratorArg,
        block: bool = True,
        timeout: float = 0,
    ) -> RequestResult:
        if self._thread is None:
            raise RuntimeError("Not started")

        if arg.num_results > self.config.max_beam_size:
            raise ValueError(f"arg.num_results {arg.num_results} is too big.")

        self.check_queue_busy()
        if not self._c_generator.submit(c_task, block):
            raise RuntimeError("Generator is busy")

        c_result = c_task.get_result(timeout)

        return RequestResult.from_cpp_stream_result(None, c_result, c_task.input_tokens_num())

    def stream_generate(
        self,
        data: Union[str, List[dict]],
        arg: GeneratorArg,
        block: bool = True,
        handler_cls=StreamHandler,
    ):
        if self._thread is None:
            raise RuntimeError("Not started")
        task, input_tokens = self._process_inputs(data, arg, True)

        self.check_queue_busy()
        if not self._c_generator.submit(task, block):
            raise RuntimeError("Generator is busy")

        h = handler_cls(task, arg, input_tokens, self._tokenizer)
        return h

    def batch_generate_c(
        self,
        input_list: List[List[int]],
        arg: Union[List[GeneratorArg], GeneratorArg],
        max_lengths=None
    ) -> List[RequestResult]:
        start = self.start()
        args = arg if isinstance(arg, list) else [arg] * len(input_list)

        # sort for better performance
        sorted_inputs = enumerate(input_list)
        if self.config.sort_by_len == 2 and max_lengths is not None:
            sorted_inputs = sorted(enumerate(input_list), key=lambda t: -max_lengths[t[0]])
        elif self.config.sort_by_len and not self.config.rag_buffer:
            sorted_inputs = sorted(enumerate(input_list), key=lambda t: len(t[1]))
        idx_tasks = []
        for i, x in sorted_inputs:
            if max_lengths is not None:
                args[i].max_length = max_lengths[i]
            idx_tasks.append((i, self.to_c_task(x, args[i])))

        tasks = [t for i, t in idx_tasks]
        # call C++ API
        sort_outs = self._c_generator.batch_search(tasks)

        # transform C++ results
        batch_outputs: List[RequestResult] = [None] * len(tasks)
        for (i, task), outputs in zip(idx_tasks, sort_outs):
            batch_outputs[i] = RequestResult.from_cpp_result(
                None, outputs, task.input_tokens_num())
        if start:
            self.stop()
        return batch_outputs

    def batch_generate(
        self,
        input_list: List[Union[str, dict]],
        arg: Union[List[GeneratorArg], GeneratorArg],
        max_in_lengths: List[int] = None,
        max_out_lengths: List[int] = None,
        prepend_input: bool = False,
    ) -> List[RequestResult]:
        batch_input = [self._encode(data) for data in input_list]
        if bool(max_in_lengths):
            if not isinstance(max_in_lengths, list):
                max_in_lengths = [max_in_lengths] * len(input_list)
            for i, max_len in enumerate(max_in_lengths):
                batch_input[i] = batch_input[i][:max_len]

            # batch_input = [x[:max_in_lengths] for x in batch_input]

        args = arg if isinstance(arg, list) else [arg.copy() for _ in input_list]

        batch_result = self.batch_generate_c(batch_input, args, max_lengths=max_out_lengths)

        for prompt, req_result in zip(input_list, batch_result):
            req_result.prompt = prompt
            for output in req_result.outputs:
                output.decode(self._tokenizer, prompt if prepend_input else None)

        return batch_result

    def start(self):
        def run_wrapper():
            try:
                self._c_generator.run()
            except Exception as e:
                if self._do_verify:
                    print("Verify max_token failed! please adjust reserved_work_mem_mb to a bigger value.")
                else:
                    print(e, flush=True)
                import signal
                os.kill(os.getpid(), signal.SIGKILL)
        if self._thread is None:
            self._thread = Thread(target=run_wrapper)
            self._thread.start()
            return True
        return False

    def stop(self):
        self._c_generator.stop()
        if self._thread is not None:
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
