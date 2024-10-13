import torch

from omni.architectures.llama import LlamaConfig
from omni.inference.inference import Inference
from omni.modules.cache import KVCache
from omni.modules.transformer import Transformer
from omni.preprocessing.tokenizer import AutoTokenizer
from omni.utils.system import auto_device


def main():
    tokenizer = AutoTokenizer.create("EleutherAI/gpt-neo-125m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_config = LlamaConfig(
        vocab_size=50258,
        seq_len=512,
        d_model=768,
        num_heads=8,
        num_kv_heads=8,
        num_layers=6,
        activation_fn="silu",
        mlp_bias=False,
        mlp_dropout=0.1,
        attention_bias=False,
        attention_dropout=0.1,
        weight_tying=False,
        pos_encoding_type="rope",
        mlp="mlp_swiglu",
        normalization="rmsnorm",
        attention="mha",
    )

    model = Transformer(model_config)

    checkpoint = torch.load("checkpoints/mha/init.ckpt", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    device = auto_device("cpu")
    inference = Inference(model, tokenizer, device=device, temperature=0, max_length=2)
    # kv_cache = KVCache(model_config, device=device)
    kv_cache = [KVCache(model_config.seq_len) for _ in range(model_config.num_layers)]

    prompt = "Once"

    print(prompt, end="", flush=True)
    import time

    start_time = time.perf_counter()

    # Generate text
    # for token in inference.generate_nonkvcache(prompt):
    #     print(tokenizer.decode([token]), end="", flush=True)

    # for token in inference.generate(prompt, kv_cache):
    #     print(tokenizer.decode([token]), end="", flush=True)

    for token in inference.generate_both(prompt, kv_cache):
        print(tokenizer.decode([token]), end="", flush=True)

    end_time = time.perf_counter()
    tokens_per_sec = 100 / (end_time - start_time)
    print(f"\n{tokens_per_sec:.2f} tokens per second")


if __name__ == "__main__":
    main()
