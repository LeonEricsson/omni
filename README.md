# omni
my collection of paper implementations and experiments. built to be modular, easy to extend, and experiment with.


### papers

- [Multi-head Latent Attention](/experiments/mla)
- [Rotary Embeddings](/omni/modules/pos_embeddings.py#L59)
- [Attention with Linear Biases](/omni/modules/pos_embeddings.py#L176)
- [Llama](/omni/architectures/llama.py)
- [GPT](/omni/architectures/gpt.py)


### Llama 30M

<img src="/experiments/basic/assets/train_loss.png" alt="Train Loss" width="300">

### todo

- [x] KV Cache
- [ ] Differential Transformer
- [ ] MLA
  - [x] KV Cache
  - [ ] Training run
  - [ ] Verify KV Cache reduction
- [x] Inference engine

### todo sides
- [ ] Swap config structure to importing modules instead
