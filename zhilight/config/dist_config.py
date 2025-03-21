import enum
from typing_extensions import TypedDict
from zhilight import C


class DistConfig(object):
    def __init__(self, tp: int = -1, dist_init_addr: str = None, nnodes: int = 1, node_rank: int = 0):
        self.tp = tp
        self.dist_init_addr = dist_init_addr
        self.nnodes = nnodes
        self.node_rank = node_rank

    @staticmethod
    def adapt(config):
        if isinstance(config, DistConfig):
            return config
        elif isinstance(config, bool):
            return DistConfig(parallel=config)
        elif isinstance(config, int):
            return DistConfig(parallel=config > 1, tp=config)
        raise ValueError("Invalid config: " + str(config))

    def to_c_config(self):
        return C.DistConfig(
            self.tp,
            self.dist_init_addr if self.dist_init_addr else "",
            self.nnodes,
            self.node_rank,
        )
    
    def __str__(self):
        return f"DistConfig(tp={self.tp}, dist_init_addr={self.dist_init_addr}, nnodes={self.nnodes}, node_rank={self.node_rank})"