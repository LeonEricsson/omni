from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float

NormalizationType = Literal["rmsnorm", "layernorm", "none"]


class RMSNorm(nn.Module):
    def __init__(
        self,
        config = None,
        dim = None,
        eps: Float = 1e-5,
    ):
        """
        Root Mean Square Normalization Layer.
        The mean accumulation is done in 32-bit precision, using the fast inverse square root.

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension to normalize over (Layer dimension)
        dim (int, optional): Dimension to normalize over. Defaults to None.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.

        References:
            - "Root Mean Square Layer Normalization" (https://arxiv.org/pdf/1910.07467)
        """
        super().__init__()

        assert config is not None or dim is not None
        norm_dim = config.d_model if config is not None else dim

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(norm_dim))

    def _irms(self, x):
        return torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        irms = self._irms(x.float()).type_as(x)
        return self.gamma * x * irms


class LayerNorm(nn.Module):
    def __init__(
        self,
        config = None,
        dim = None,
        eps: Float = 1e-5,
    ):
        """
        Layer Normalization.
        The mean and variance accumulation is done in 32-bit precision.

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension to normalize over (Layer dimension)
        dim (int, optional): The size of the dimension which is normalized over, match 'x' in forward pass.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.

        References:
            - "Layer Normalization" (https://arxiv.org/abs/1607.06450)
        """
        super().__init__()

        assert config is not None or dim is not None
        norm_dim = config.d_model if config is not None else dim
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(norm_dim))
        self.beta = nn.Parameter(torch.zeros(norm_dim))

    def _norm_scaling(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x):
        normalized = self._norm_scaling(x.float()).type_as(x)
        return self.gamma * normalized + self.beta


NORM_MAP = {
    "rmsnorm": RMSNorm,
    "layernorm": LayerNorm,
    "none": nn.Identity,
}
