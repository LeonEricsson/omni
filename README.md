# omni
my collection of paper implementations and experiments. built to be modular, easy to extend, and experiment with.


### papers

- [Multi-head Latent Attention](/experiments/mla)
- [Rotary Embeddings](/omni/modules/pos_embeddings.py#L59)
- [Attention with Linear Biases](/omni/modules/pos_embeddings.py#L176)
- [Llama](/omni/architectures/llama.py)
- [GPT](/omni/architectures/gpt.py)


### todo

- [ ] Train 20M small model on >1B tokens
- [x] Inference engine
- [x] KV Cache
- MLA
  - [x] KV Cache
  - [ ] Training run
  - [ ] Verify KV Cache reduction
- [ ] Differential Transformer
- [ ] Swap config structure to importing modules instead
- [ ] Compare GQA run with the optimized forward function.
- [ ] GQA infernece check, i think the kv cache should come before
- [ ] the k, v repetition?
