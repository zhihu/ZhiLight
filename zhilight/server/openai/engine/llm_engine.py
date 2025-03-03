from typing import List, Optional, Union, Dict

from zhilight.server.openai.lora.request import LoRARequest
from zhilight.server.openai.basic.config import EngineConfig
from zhilight.server.openai.engine.arg_utils import EngineArgs
from zhilight.server.openai.basic.logger import init_logger
from zhilight.server.openai.basic.outputs import RequestOutput
from zhilight.server.openai.basic.sampling_params import SamplingParams
from zhilight import LLaMA
from zhilight.dynamic_batch import DynamicBatchGenerator, GeneratorArg
from zhilight.models.auto_model import AutoModel, is_customized_model

logger = init_logger(__name__)

class LLMEngine:
    def __init__(
        self,
        engine_config: EngineConfig,
        log_stats: bool,
    ) -> None:
        logger.info(f"engine config => {engine_config}")
        self.log_stats = log_stats

        # Load model
        if is_customized_model(engine_config.model_config.get("model_type")):
            self._instance = AutoModel.from_pretrained(engine_config.model_path)
        else:
            self._instance = LLaMA(
                model_path=engine_config.model_path,
                model_config=engine_config.model_config,
                quant_config=engine_config.quant_config,
                parallel=engine_config.enable_tensor_parallel,
            )
            if engine_config.is_cpm_directory_struct:
                assert not engine_config.use_safetensors, "not support safetensors for old cpm load method."
                self._instance.load_model_pt(engine_config.model_file)
            else:
                if engine_config.use_safetensors:
                    self._instance.load_model_safetensors(engine_config.model_path)
                else:
                    self._instance.load_model_pt(engine_config.model_path)
        
        # Create dyn generator
        self.dyn_generator = DynamicBatchGenerator(
            engine_config.dyn_batch_config,
            self._instance,
        )
        self.dyn_generator.start()
        
        self.engine_config = engine_config

    def stop(self):
        self.dyn_generator.stop()
        logger.info("dyn generator exit.")

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        # Create the LLM engine.
        engine = cls(engine_config,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[Dict[str, str]]],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        beam_size = 1
        temperature = 1.0
        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        if sampling_params.use_beam_search and sampling_params.best_of is not None and sampling_params.best_of > 0:
            beam_size = sampling_params.best_of
        if sampling_params.temperature < 1e-5:
            beam_size = 1
            top_p = 1.0
            top_k = 0
        else:
            temperature = sampling_params.temperature
        if top_k == -1:
            top_k = 0
        
        arg = GeneratorArg(
            beam_size = beam_size,
            max_length = sampling_params.max_tokens if sampling_params.max_tokens is not None else 512,
            repetition_penalty = sampling_params.repetition_penalty,
            ngram_penalty = 1.0,
            seed = sampling_params.seed,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            presence_penalty = sampling_params.presence_penalty,
            num_results = sampling_params.n,
        )

        logger.info(f"request={request_id} dyn_arg=<beam_size={arg.beam_size}, "
                    f"max_length={arg.max_length}, repetition_penalty={arg.repetition_penalty}, "
                    f"ngram_penalty={arg.ngram_penalty}, seed={arg.seed}, temperature={arg.temperature}, "
                    f"top_p={arg.top_p}, top_k={arg.top_k}, presence_penalty={arg.presence_penalty}, "
                    f"num_results={arg.num_results}>")

        stream = self.dyn_generator.stream_generate(prompt, arg)
        return stream

    def get_engine_config(self) -> EngineConfig:
        return self.engine_config