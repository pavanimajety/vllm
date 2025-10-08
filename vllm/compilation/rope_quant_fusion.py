from abc import ABC, abstractmethod

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


class RopeQuantPattern(ABC):
    """
    The base class for Rope+Quant fusions.
    Should not be used directly.
    """

    def __init__(self, layer: RotaryEmbedding):
        return
