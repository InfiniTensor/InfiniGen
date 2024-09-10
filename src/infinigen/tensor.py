from enum import Enum

import ninetoothed


class Tensor(ninetoothed.Tensor):
    def __init__(
        self,
        ndim=None,
        shape=None,
        dtype=None,
        strides=None,
        memory=None,
        original=None,
    ):
        super().__init__(
            ndim=ndim,
            shape=shape,
            dtype=dtype if dtype is not None else DataType.FLOAT32,
            strides=strides,
            original=original,
        )

        self.memory = memory if memory is not None else MemoryType.GLOBAL
        self.producer = None
        self.consumers = []
        self.uses = 0

    def __repr__(self):
        return f"[{self.name}; {', '.join(str(size) for size in self.shape)}; {', '.join(str(stride) for stride in self.strides)}]"


class DataType(Enum):
    FLOAT32 = "float32"


class MemoryType(Enum):
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER_FILE = "register_file"
