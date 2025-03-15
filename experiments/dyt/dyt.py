import torch
import torch.nn as nn
from jaxtyping import Float


class DyT(nn.Module):
    def __init__(
        self,
        config = None,
        eps: Float = 1e-5,
    ):
        """
        Root Mean Square Normalization Layer.
        The mean accumulation is done in 32-bit precision, using the fast inverse square root.

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension to normalize over
            - init_alpha (float): 
        dim (int, optional): Dimension to normalize over. Defaults to None.

        References:
            - "Transformers without Normalization" (https://arxiv.org/abs/2503.10622)
        """
        super().__init__()

        self.alpha = nn.Parameter(config.init_alpha * torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x  + self.beta