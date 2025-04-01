"""
This file contains the command line arguments for the CPM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
import dataclasses
from dataclasses import dataclass

from zhilight.server.openai.engine.arg_utils import AsyncEngineArgs
from zhilight.server.openai.entrypoints.serving_engine import LoRA
from zhilight.server.openai.entrypoints.preparse_cli_args import add_preparse_argmuents


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split('=')
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


@dataclass
class OpenAIServingArgs:
    response_role: str
    enable_reasoning: bool
    reasoning_parser: str

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--enable-reasoning",
            action="store_true",
            help="Enable reasoning for the model. If enabled, the model will be able to generate reasoning content."
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=["deepseek-r1"],
            default=None,
            help="Select the reasoning parser to use. Required if `--enable-reasoning` is enabled."
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'OpenAIServingArgs':
        return cls(
            response_role=args.response_role,
            enable_reasoning=args.enable_reasoning,
            reasoning_parser=args.reasoning_parser,
        )


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="CPM-Server OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8080, help="port number")
    parser.add_argument("--api-key",
                        type=str,
                        default=None,
                        help="If provided, the server will require this key "
                        "to be presented in the header.")
    
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")

    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="log level for uvicorn")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, CPM-Server will add it to the server "
        "using @app.middleware('http'). "
        "If a class is provided, CPM-Server will add it to the server "
        "using app.add_middleware(). ")

    add_preparse_argmuents(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    parser = OpenAIServingArgs.add_cli_args(parser)
    return parser


def validate_parsed_serve_args(args: argparse.Namespace):
    if args.enable_reasoning and not args.reasoning_parser:
        raise TypeError("Error: `--reasoning-parser` is required if `--enable-reasoning` is enabled.")