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
            # hidden_dim = 4 * int(2 * config.d_model / 3)
            # hidden_dim = 4 * (
            #     (hidden_dim + 4 - 1) // 4
            # )  # ensure hidden_dim is divisible by 4
            hidden_dim = 4 * config.d_model

        self.W_uv = nn.Linear(config.d_model, 2 * hidden_dim, bias=config.mlp_bias)
        self.W_o = nn.Linear(hidden_dim, config.d_model, bias=config.mlp_bias)

        self.dropout = (
            nn.Dropout(config.mlp_dropout) if config.mlp_dropout else lambda x: x
        )

        s_uv_init = 1.0
        s_uv_scale = 1.0
        self.s_uv = nn.Parameter(
            s_uv_scale * torch.ones(2 * hidden_dim, dtype=torch.float32)
        )
        self.register_buffer("s_uv_init", torch.tensor(s_uv_init))
        self.register_buffer("s_uv_scale", torch.tensor(s_uv_scale))

        self.register_buffer("sqrt_d_model", torch.tensor(config.d_model**0.5))

    def normalize_weights(self):
        self.W_uv.weight.data = F.normalize(self.W_uv.weight.data, dim=-1)
        self.W_o.weight.data = F.normalize(self.W_o.weight.data, dim=-1)

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        uv = self.W_uv(x)

        suv = self.s_uv * ((self.s_uv_init / self.s_uv_scale) * self.sqrt_d_model)
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)

        x_mlp = u * F.silu(v)  # SwiGLU

        x_mlp = self.dropout(self.W_o(x_mlp))

        return x_mlp
