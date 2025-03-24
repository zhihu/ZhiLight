# CPM-Server OpenAI Interface based on vLLM code and zhilight engine
# Author: wangjingjing@zhihu.com, vLLM team
#

from zhilight.server.openai.entrypoints.preparse_cli_args import preparse_args
from zhilight.server.openai.basic.logger import init_logger
from zhilight.server.openai.basic.utils import (
                                    parse_zhilight_version,
                                    register_environs,
                                    get_options_info,
                                    make_async,
                                    force_install_packages)
engine_version = None
logger = init_logger(__name__)


args, _ARGV_ = preparse_args()
if args is not None:
    engine_version = parse_zhilight_version(args.zhilight_version)
    force_install_packages(args.pip)
    register_environs(args.environ)

import os
import asyncio
from contextlib import asynccontextmanager
import torch

from prometheus_client import make_asgi_app
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, Response

import zhilight.server
from zhilight.server.openai.engine.arg_utils import AsyncEngineArgs
from zhilight.server.openai.engine.async_llm_engine import AsyncLLMEngine
from zhilight.server.openai.entrypoints.protocol import (CompletionRequest,
                                              ChatCompletionRequest,
                                              ErrorResponse)
from zhilight.server.openai.basic.utils import (
                                    parse_zhilight_version,
                                    register_environs,
                                    get_options_info,
                                    make_async,
                                    force_install_packages)
from zhilight.server.openai.entrypoints.cli_args import make_arg_parser
from zhilight.server.openai.entrypoints.serving_chat import OpenAIServingChat
from zhilight.server.openai.entrypoints.serving_completion import OpenAIServingCompletion
from zhilight.server.openai.entrypoints.middleware import add_middleware

TIMEOUT_KEEP_ALIVE = 5  # seconds

parser = None
openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None

def parse_args():
    global parser
    parser = make_arg_parser()
    return parser.parse_args(args=_ARGV_)

# 定时输出 stats log
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield
    # stop zhilight engine
    await engine.stop()

app = fastapi.FastAPI(lifespan=lifespan)
fake_app = fastapi.FastAPI(lifespan=lifespan)


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
@fake_app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
@app.get("/api/check_health")
@fake_app.get("/health")
@fake_app.get("/api/check_health")
async def health() -> Response:
    """Health check."""
    # TODO: check model status
    # await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": engine_version}
    return JSONResponse(content=ver)

@app.get("/options")
async def get_options():
    options = await make_async(get_options_info)(parser)
    return JSONResponse(content=options)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())

# python -m zhilight.server.openai.entrypoints.api_server [Options]
if __name__ == "__main__":

    args = parse_args()

    logger.info(f"ZhiLight OpenAI-Compatible Server version {engine_version}.")
    if args.node_rank > 0:
        app = fake_app

    if args.enable_prefix_caching: # FixME
        os.environ["enable_prompt_caching"] = "1"

    add_middleware(app, args)

    args.api_key = "*"
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model_path

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    openai_serving_chat = OpenAIServingChat(engine, served_model, args.response_role)
    openai_serving_completion = OpenAIServingCompletion(engine, served_model)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
