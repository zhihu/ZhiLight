import asyncio
import os
import copy
import time
import json
from fastapi import Request
from typing import (Callable, Dict, List, Optional, Type, Union, AsyncIterator)

from zhilight.server.openai.lora.request import LoRARequest
from zhilight.server.openai.basic.config import EngineConfig
from zhilight.server.openai.basic.utils import make_async
from zhilight.server.openai.basic.sequence import RequestMetrics
from zhilight.server.openai.engine.metrics import StatLogger, Stats
from zhilight.server.openai.basic.logger import init_logger
from zhilight.server.openai.basic.outputs import RequestOutput, CompletionOutput
from zhilight.server.openai.basic.sampling_params import SamplingParams
from zhilight.server.openai.engine.arg_utils import AsyncEngineArgs
from zhilight.server.openai.engine.llm_engine import LLMEngine
from zhilight.dynamic_batch import StreamHandler, StreamResultType

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("CPM_ENGINE_ITERATION_TIMEOUT_S", "60"))
_DELAY_HANDLER_DELETE_SEC = int(
    os.environ.get("CPM_DELAY_HANDLER_DELETE_SEC", "120"))
_LOCAL_LOGGING_INTERVAL_SEC = 5

class AsyncEngineDeadError(RuntimeError):
    pass

def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! ")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e

class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str, handler: StreamHandler, arrival_time: float) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self.arrival_time = arrival_time
        self._finished = False
        self._canceled = False
        self._handler = handler

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    def cancel(self) -> None:
        self._canceled = True

    @property
    def finished(self) -> bool:
        return self._finished
    
    @property
    def canceled(self) -> bool:
        return self._canceled

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    
class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def add_request_async(
        self,
        request_id: str,
        prompt: Union[str, List[Dict[str, str]]],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")

        handler = await make_async(self.add_request)(
            request_id,
            prompt,
            sampling_params,
            prompt_token_ids,
            arrival_time,
            lora_request, 
        )

        return handler

    async def check_health_async(self) -> None:
        self.check_health()


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine
    _stat = Stats(time.time(), 0, 0, 0, 0, [], [], [])

    def __init__(self,
                 engine_config: EngineConfig,
                 *args,
                 log_requests: bool = True,
                 log_stats: bool = False,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._engine_class(engine_config = engine_config, log_stats = log_stats)

        self._errored_with: Optional[BaseException] = None
        self.stat_logger = None
        if log_stats:
            self.stat_logger = StatLogger(
                local_interval = _LOCAL_LOGGING_INTERVAL_SEC,
                labels = dict(model_name = engine_config.model_path))

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = True) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Create the async LLM engine.
        engine = cls(engine_config,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    async def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[Dict[str, str]]],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncStream:
        if self.log_requests:
            logger.info(f"Received request {request_id}: "
                        f"prompt: {prompt!r}, "
                        f"sampling_params: {sampling_params}, "
                        f"prompt_token_ids: {prompt_token_ids}, "
                        f"lora_request: {lora_request}.")

        if arrival_time is None:
            arrival_time = time.time()

        handler = await self.engine.add_request_async(
            request_id,
            prompt,
            sampling_params,
            prompt_token_ids,
            arrival_time,
            lora_request,
        )

        stream = AsyncStream(request_id, handler, arrival_time)
        asyncio.create_task(self.step_loop(prompt, sampling_params, handler, stream))

        return stream

    async def step_loop(self, prompt: Union[str, List[Dict[str, str]]], sampling_params: SamplingParams, handler: StreamHandler, stream: AsyncStream):
        first = True
        finished = False
        tm_beg = time.time()
        last_iter_time = tm_beg
        stats = RequestMetrics(stream.arrival_time, 0)
        current_output = None
        while not stream.canceled:
            output = await make_async(handler.decode_stream_res)(handler.get_result(), increasing = False)
            current_iter_time = time.time()
            iter_cost = current_iter_time - last_iter_time
            if first:
                stats.first_token_time = current_iter_time
                stats.first_scheduled_time = handler.task.get_first_schedule_ts() * 1e-6 
                stats.time_in_queue = handler.task.get_time_in_queuegs() * 1e-6
                stats.input_tokens_num = handler.input_tokens_num
                first = False
            stats.output_tokens_num = sum(handler.output_tokens_nums)
            if output[0] == StreamResultType.Final:
                stats.finished_time = time.time()
                finished = True
            else:
                stats.last_token_time = iter_cost
                last_iter_time = current_iter_time
            # TODO: support multi results
            current_output = RequestOutput(
                request_id = stream.request_id,
                prompt = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False),
                prompt_token_ids_num = handler.input_tokens_num,
                prompt_logprobs = None,
                outputs = [
                    CompletionOutput(
                        0,
                        output[2],
                        handler.output_tokens_nums[0],
                        output[3],
                        None,
                        None if not finished else "stop", # TODO, set real reason
                        None
                    ),
                ],
                finished = finished,
                metrics = copy.copy(stats),
            )
            stream.put(current_output)
            if finished:
                if self.log_requests:
                    logger.info(f"Final Response {current_output}, params={sampling_params}, cost={time.time()-tm_beg:.3f}s")
                break
        if stream.canceled:
            await make_async(handler.cancel)()
            logger.warn(f"Request={stream.request_id} canceled. Final Response {current_output}, params={sampling_params}, cost={time.time()-tm_beg:.3f}s")
        stream.finish()

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        sampling_params: SamplingParams,
        raw_request: Request,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[RequestOutput]:
        # Preprocess the request.
        arrival_time = time.time()
        generated_tokens_num = 0
        self._stat.num_total += 1
        do_running = False
        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
            )
            async for request_output in stream:
                # deal disconnect
                if await raw_request.is_disconnected():
                    stream.cancel()
                    raise StopAsyncIteration("Client disconnected.")
                # collect metrics
                first_token_cost = request_output.metrics.first_token_time
                if first_token_cost is not None:
                    self._stat.time_to_first_tokens.append(first_token_cost)
                    self._stat.num_prompt_tokens += request_output.metrics.input_tokens_num
                    self._stat.num_running += 1
                    do_running = True
                self._stat.time_per_output_tokens.append(request_output.metrics.last_token_time)
                generated_tokens_num = request_output.metrics.output_tokens_num
                # output
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            raise e
        finally:
            self._stat.time_e2e_requests.append(time.time() - arrival_time)
            self._stat.num_generation_tokens += generated_tokens_num
            await self.do_log_stats()
            if do_running:
                self._stat.num_running -= 1
            self._stat.num_total -= 1

    async def do_log_stats(self) -> None:
        if self.stat_logger is not None:
            self.stat_logger.log(self._stat)

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        # Todo, check dead loop
        pass

    async def get_engine_config(self) -> EngineConfig:
        return self.engine.get_engine_config()
    
    async def stop(self) -> None:
        await make_async(self.engine.stop)()