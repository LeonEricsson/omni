from typing import Literal

import torch.nn.functional as F


ActivationFunction = Literal[
    "relu", "gelu", "tanh", "silu", "sigmoid", "identity", "none"
]

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "silu": F.silu,
    "sigmoid": F.sigmoid,
    "identity": lambda x: x,
    "none": lambda x: x,
}
