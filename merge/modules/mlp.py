import torch.nn as nn
import torch.nn.functional as F

from merge.modules.activations import ACT2FN
from merge.modules.config import TransformerConfig


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        MLP - Transformer specific with 2 linear transformations.
        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): Input dimension
            - hidden_dim (int, optional): Hidden dimension. If None, computed as 4 * (2/3 * hidden_dim)
            - activation_fn (str): Activation function name. Defaults to "relu"
            - dropout (float, optional): Dropout probability. Defaults to None
            - bias (bool): Whether to use bias. Defaults to True
        """
        super().__init__()
        hidden_dim = config.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * int(2 * hidden_dim / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.dropout = nn.Dropout(config.dropout) if config.dropout else lambda x: x

    def forward(self, x):
        x = self.up(x)
        x = self.activation_fn(x)
        x = self.down(x)
        x = self.dropout(x)
        return x


class MLPSwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        SwiGLU Feed Forward Network - Transformer specific (2 layers, 1 of which is gated).

        Args:
        config (TransformerConfig): Configuration dataclass containing:
            - d_model (int): The input dimension.
            - hidden_dim (int, optional): Hidden dimension. If None, computed as 4 * (2/3 * hidden_dim)
            - dropout (float, optional): Dropout probability. Defaults to None
            - bias (bool): Whether to use bias. Defaults to False

        References:
            - "Gated Linear Units" (https://arxiv.org/pdf/2002.05202v1)
        """
        super().__init__()
        hidden_dim = config.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * int(2 * hidden_dim / 3)
            hidden_dim = 4 * (
                (hidden_dim + 4 - 1) // 4
            )  # ensure hidden_dim is divisible by 4

        self.up = nn.Linear(config.d_model, hidden_dim, bias=config.mlp_bias)
        self.gate = nn.Linear(config.d_model, hidden_dim, bias=config.mlp_bias)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=config.mlp_bias)
        self.dropout = (
            nn.Dropout(config.mlp_dropout) if config.mlp_dropout else lambda x: x
        )

    def forward(self, x):
        x = self.up(x) * F.silu(self.gate(x))  # SwiGLU
        x = self.down(x)
        x = self.dropout(x)
        return x


MLP_MAP = {
    "mlp": MLP,
    "mlp_swiglu": MLPSwiGLU,
}
