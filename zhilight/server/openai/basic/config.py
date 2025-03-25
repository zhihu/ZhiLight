from dataclasses import dataclass
from zhilight.dynamic_batch import DynamicBatchConfig
from zhilight.quant import QuantConfig
from zhilight.config.dist_config import DistConfig
@dataclass
class EngineConfig:
    model_path: str
    model_file: str # for compatible
    vocab_file: str # for compatible
    is_cpm_directory_struct: bool
    use_safetensors: bool
    model_config: dict
    dyn_batch_config: DynamicBatchConfig
    quant_config: QuantConfig = None
    dist_config: DistConfig = None
    memory_limit: int = 0
    is_chatml: bool = False
    max_model_len: int = 8192

    def __post_init__(self):
        pass