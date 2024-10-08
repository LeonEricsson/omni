import torch

from omni.architectures.llama import LlamaConfig
from omni.modules.transformer import Transformer
from omni.preprocessing.tokenizer import AutoTokenizer
from omni.utils.system import auto_device
from omni.inference.inference import Inference
from omni.modules.cache import KVCache

def main():
    tokenizer = AutoTokenizer.create("EleutherAI/gpt-neo-125m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    llama_config = LlamaConfig(
        vocab_size=50258,
        seq_len=512,
        d_model=256,
        num_heads=8,
        num_kv_heads=8,
        num_layers=4,
        rope_theta=0.1,
        norm_eps=1e-6,
        activation_fn="silu",
        mlp_bias=False,
        mlp_dropout=0.0,
        attention_bias=False,
        attention_dropout=0.0,
        pos_encoding_type="rope",
        mlp="mlp_swiglu",
        normalization="rmsnorm",
        attention="gqa",
    )

    model = Transformer(llama_config)

    checkpoint = torch.load("checkpoints/llama-30M_20250123_104138/init.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    device = auto_device()
    inference = Inference(model, tokenizer, device=device, temperature=0, max_length=100)
    kv_cache = KVCache(llama_config, device=device)

    prompt = "Once upon a time "

    print(prompt, end="", flush=True)
    import time
    start_time = time.perf_counter()

    # Generate text
    for token in inference.generate_nonkvcache(prompt):
        print(tokenizer.decode([token]), end="", flush=True)
    
    # for token in inference.generate(prompt, kv_cache):
    #     print(tokenizer.decode([token]), end="", flush=True)

    end_time = time.perf_counter()
    tokens_per_sec = 100 / (end_time - start_time)
    print(f"\n{tokens_per_sec:.2f} tokens per second")

if __name__ == "__main__":
    main()