import os
import json
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional

from zhilight.server.openai.basic.config import EngineConfig
from zhilight.loader import LLaMALoader
from zhilight.dynamic_batch import DynamicBatchConfig
from zhilight.quant import QuantConfig, QuantType


@dataclass
class EngineArgs:
    """Arguments for ZhiLight engine."""
    model_path: str
    max_model_len: Optional[int] = 8192
    disable_flash_attention: bool = False
    disable_tensor_parallel: bool = False
    enable_prefix_caching: bool = False
    disable_log_stats: bool = False
    quantization: Optional[str] = None
    dyn_max_batch_size: int = 8
    dyn_max_beam_size: int = 4
    ignore_eos: bool = False
    enable_cpm_chat: bool = False

    def __post_init__(self):
        pass

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for ZhiLight engine."""

        # Model arguments
        parser.add_argument(
            '--model-path',
            type=str,
            default='/mnt/models',
            help='path of the zhilight model to use')

        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')

        parser.add_argument(
            '--disable-flash-attention',
            action='store_true',
            help='If specified, turn off flash attention speedup.')

        parser.add_argument(
            '--enable-cpm-chat',
            action='store_true',
            help='If specified, use chatml tokenizer for cpm model.')

        parser.add_argument(
            '--disable-tensor-parallel',
            action='store_true',
            help='If specified, use tensor parallel, otherwise, use pipeline parallel.')

        parser.add_argument('--enable-prefix-caching',
                            action='store_true',
                            help='Enables automatic prefix caching')


        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')

        parser.add_argument('--quantization',
                            '-q',
                            type=str,
                            choices=['AbsMax', 'AutoInt8', 'Int4', 'AutoInt4', 'GPTQ'],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')

        parser.add_argument('--dyn-max-batch-size',
                            type=int,
                            default=EngineArgs.dyn_max_batch_size,
                            help='Maximum batch size for dyn batch.')

        parser.add_argument('--dyn-max-beam-size',
                            type=int,
                            default=EngineArgs.dyn_max_beam_size,
                            help='Maximum beam size for dyn batch.')
                            
        parser.add_argument('--ignore-eos',
                            action='store_true',
                            help='Ignore eos for pressure test.')

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
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Excluded attributes
        excluded_attrs = []
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls) if attr.name not in excluded_attrs]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(self) -> EngineConfig:
        # Model Config
        model_config = LLaMALoader.load_llama_config(self.model_path)
        if model_config.get("model_type", "") == "cpm_dragonfly":
            model_config["new_vocab"] = False

        model_file, use_safetensors = self._parse_model_file()
        vocab_file = f"{self.model_path}/vocabs.txt"
        is_cpm_dir_struct = self._is_cpm_directory_struct()
        if is_cpm_dir_struct:
            with open(f"{self.model_path}/config.json", "r") as f:
                model_config = json.load(f)
            if "new_vocab" not in model_config:
                model_config["new_vocab"] = True
            model_config["scale_weights"] = True
            model_config["weight_transposed"] = False

        # Dyn Batch Config
        dyn_batch_config = DynamicBatchConfig(
            max_batch = self.dyn_max_batch_size,
            max_beam_size = self.dyn_max_beam_size,
            eos_id = 2,
            keep_eos = False,
            ignore_eos = self.ignore_eos,
            high_precision = -1,
            max_total_token = self.max_model_len,
            flash_attention = not self.disable_flash_attention,
        )

        # Quant Config
        q_type = QuantType.NoQuant
        if self.quantization is not None:
            _map = {}
            for key, val in QuantType.__members__.items():
                _map[key.lower()] = val
            key = self.quantization.lower()
            if key in _map:
                q_type = _map[key]
            
        return EngineConfig(
            model_path = self.model_path,
            model_file = model_file,
            vocab_file = vocab_file,
            is_cpm_directory_struct = is_cpm_dir_struct,
            use_safetensors = use_safetensors,
            model_config = model_config,
            dyn_batch_config = dyn_batch_config,
            quant_config = QuantConfig(type = q_type),
            enable_tensor_parallel = not self.disable_tensor_parallel,
            max_model_len = self.max_model_len,
            is_chatml = self.enable_cpm_chat,
        )

    def _is_cpm_directory_struct(self) -> bool:
        config_file = f"{self.model_path}/config.json"
        vocab_file = f"{self.model_path}/vocabs.txt"
        tokenizer_file = f"{self.model_path}/tokenizer.model"
        return os.path.isfile(config_file) and \
            os.path.isfile(vocab_file) and not os.path.isfile(tokenizer_file)

    def _parse_model_file(self):
        for path, _, files in os.walk(self.model_path):
            for file in files:
                if file.endswith(".pt") or file.endswith(".bin"):
                    return os.path.join(path, file), False
                if file.endswith(".safetensors"):
                    return os.path.join(path, file), True

        raise RuntimeError(f"no supported model file in {self.model_path}. [*.pt, *.bin, *.safetensors]")


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous ZhiLight engine."""
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)

        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='disable logging requests')
        parser.add_argument('--max-log-len',
                            type=int,
                            default=None,
                            help='max number of prompt characters or prompt '
                            'ID numbers being printed in log. '
                            'Default: unlimited.')
        return parser
