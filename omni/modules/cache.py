import torch
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Int

from omni.modules.config import TransformerConfig



class KVCache:
    def __init__(self, config: TransformerConfig, device: str = None, dtype: torch.dtype = None):
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.max_seq_len = config.seq_len
        self.device = device
        self.dtype = dtype
        
        head_dim = config.d_model // config.num_kv_heads
        cache_shape = (1, self.num_layers, self.num_kv_heads, self.max_seq_len, head_dim)

        self.k = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v = torch.zeros(cache_shape, device=device, dtype=dtype)
        
        self.cache_lengths = torch.zeros(self.num_layers, device=device, dtype=torch.int16)

    def forward(self, layer_idx: int, k: Float[Tensor, "batch num_heads seq head_dim"], v: Float[Tensor, "batch num_heads seq head_dim"]):
        """Update the cache for a single layer, handling overflow by rolling."""
        cache_len = self.cache_lengths[layer_idx].item()
        new_len = k.size(2)
        max_len = self.max_seq_len

        if cache_len + new_len > max_len:
            overflow = cache_len + new_len - max_len
            self.k[:, layer_idx, :, :-overflow, :] = self.k[:, layer_idx, :, overflow:, :]
            self.v[:, layer_idx, :, :-overflow, :] = self.v[:, layer_idx, :, overflow:, :]
            cache_len = max_len - new_len


        self.k[:, layer_idx, :, cache_len:cache_len + new_len, :] = k
        self.v[:, layer_idx, :, cache_len:cache_len + new_len, :] = v

        self.cache_lengths[layer_idx] = min(cache_len + new_len, max_len)

        return self.k[:, layer_idx, :, :self.cache_lengths[layer_idx], :], \
               self.v[:, layer_idx, :, :self.cache_lengths[layer_idx], :]
