import enum
from typing_extensions import TypedDict


class DistConfig(object):
    def __init__(self, parallel, tp=0):
        self.parallel: bool = parallel
        self.tp: int = tp

    @staticmethod
    def adapt(config):
        if isinstance(config, DistConfig):
            return config
        elif isinstance(config, bool):
            return DistConfig(parallel=config)
        elif isinstance(config, int):
            return DistConfig(parallel=config > 1, tp=config)
        raise ValueError("Invalid config: " + str(config))
