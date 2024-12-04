import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from torch import Tensor

class nMLPSwiGLU(nn.Module):
    def __init__(self, config):
        """
        SwiGLU Feed Forward Network - Transformer specific (2 layers, 1 of which is gated).

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): The input dimension.
            - hidden_dim (int, optional): Hidden dimension. If None, computed as 4 * (2/3 * d_model)
            - dropout (float, optional): Dropout probability. Defaults to None
            - bias (bool): Whether to use bias. Defaults to False

        References:
            - "Gated Linear Units" (https://arxiv.org/pdf/2002.05202v1)
        """
        super().__init__()
        hidden_dim = config.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * int(2 * config.d_model / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(config.d_model, hidden_dim, bias=config.mlp_bias)
        self.gate = nn.Linear(config.d_model, hidden_dim, bias=config.mlp_bias)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=config.mlp_bias)
        self.dropout = (
            nn.Dropout(config.mlp_dropout) if config.mlp_dropout else lambda x: x
        )

        self.s_u = nn.Parameter(torch.ones(hidden_dim))
        self.s_v = nn.Parameter(torch.ones(hidden_dim))
        
        self.register_buffer("s_u_scale", torch.tensor(config.d_model**-0.5))

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        u = self.up(x)              
        v = self.gate(x)            

        u = u * self.s_u
        v = v * (self.s_v * self.s_u_scale)

        x_mlp = u * F.silu(v) # SwiGLU
        
        x_mlp = self.dropout(self.down(x_mlp))

        return x_mlp