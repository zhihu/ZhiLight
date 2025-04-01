import time
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from zhilight.server.openai.basic.logger import init_logger
from zhilight.server.openai.basic.utils import get_traceid
from zhilight.server.openai.engine.async_llm_engine import AsyncLLMEngine
from zhilight.server.openai.entrypoints.cli_args import OpenAIServingArgs
from zhilight.server.openai.entrypoints.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from zhilight.server.openai.basic.outputs import RequestOutput
from zhilight.server.openai.entrypoints.serving_engine import OpenAIServing, LoRA
from zhilight.server.openai.entrypoints.reasoning_parser import ReasoningParser


logger = init_logger(__name__)

class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 serving_args: OpenAIServingArgs,
                 lora_modules: Optional[List[LoRA]] = None,
                 chat_template=None,
                 ):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=lora_modules)
        self._load_chat_template(chat_template)
        self.response_role = serving_args.response_role
        self.enable_reasoning = serving_args.enable_reasoning
        self.reasoning_parser = serving_args.reasoning_parser
    

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        try:
            '''
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
            '''
            prompt = request.messages
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        request_id = f"cmpl-{get_traceid(raw_request)}"
        try:
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
        except Exception as e:
            return self.create_error_response(str(e))
        result_generator = self.engine.generate(prompt, sampling_params,
                                                raw_request, request_id, None,
                                                lora_request)
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, raw_request, result_generator, request_id)
        try:
            return await self.chat_completion_full_generator(
                request, raw_request, result_generator, request_id)
        except Exception as e:
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:
        model_name = request.model
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True
        should_stream_with_reasoning_parsing =  self._should_stream_with_reasoning_parsing(request)
        if should_stream_with_reasoning_parsing:
            reasoner_parser = ReasoningParser(model_type=self.reasoning_parser)
        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                res: RequestOutput
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        delta = DeltaMessage(role=role, content=None)
                        if should_stream_with_reasoning_parsing:
                            delta = DeltaMessage(role=role, reasoning_content=None)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta,
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if request.messages and isinstance(
                                request.messages,
                                list) and request.messages[-1].get(
                                    "content") and request.messages[-1].get(
                                        "role") == role:
                            last_msg_content = request.messages[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    top_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs:
                        logprobs = self._create_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=top_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            initial_text_offset=len(previous_texts[i]),
                        )
                    else:
                        logprobs = None

                    delta_text = output.text[len(previous_texts[i]):]
                    delta = DeltaMessage(content=delta_text)
                    if should_stream_with_reasoning_parsing:
                        reasoning_content, _ = reasoner_parser.parse_stream_chunk(delta_text)
                        if reasoning_content:
                            delta = DeltaMessage(reasoning_content=reasoning_content)
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta,
                            logprobs=logprobs,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        num_prompt_tokens = res.prompt_token_ids_num
                        num_generated_tokens = output.token_ids_num
                        final_usage = UsageInfo(
                            prompt_tokens = num_prompt_tokens,
                            completion_tokens = num_generated_tokens,
                            total_tokens = num_prompt_tokens + num_generated_tokens,
                        )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if final_usage is not None:
                            chunk.usage = final_usage
                        data = chunk.model_dump_json(exclude_unset=True,
                                                     exclude_none=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True
        except Exception as e:
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = request.model
        created_time = int(time.time())
        final_res: RequestOutput = None

        async for res in result_generator:
            final_res = res
        assert final_res is not None

        choices = []
        should_stream_with_reasoning_parsing =  self._should_stream_with_reasoning_parsing(request)
        if should_stream_with_reasoning_parsing:
            reasoner_parser = ReasoningParser(model_type=self.reasoning_parser)

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            top_logprobs = output.logprobs

            if request.logprobs:
                logprobs = self._create_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None
            
            message = ChatMessage(role=role, content=output.text)
            if should_stream_with_reasoning_parsing:
                reasoning_content, content = reasoner_parser.parse_non_stream(output.text)
                if reasoning_content:
                    message = ChatMessage(role=role, content=content, reasoning_content=reasoning_content)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = res.prompt_token_ids_num
        num_generated_tokens = sum(
            output.token_ids_num for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def _load_chat_template(self, chat_template):
        pass
    

    def _should_stream_with_reasoning_parsing(self,
                                            request: ChatCompletionRequest):
        """
        if reasoning is enabled and a reasoning parser is configured, we should parse the outputs.
        """
        return self.enable_reasoning and request.separate_reasoning