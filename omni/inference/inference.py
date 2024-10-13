import torch
import torch.nn as nn

from omni.modules.transformer import KVCache


class Inference:
    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        max_length=50,
        temperature=1.0,
        top_k=500,
        top_p=0.95,
        min_p=0.05,
    ):
        assert temperature >= 0.0
        assert top_k >= 0
        assert top_p >= 0.0 and top_p <= 1.0
        assert min_p >= 0.0 and min_p <= 1.0

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k if top_k > 0 else tokenizer.vocab_size
        self.top_p = top_p
        self.min_p = min_p

        self.model.to(self.device)
        self.model.eval()

    def _sample(self, logits):
        if self.temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / self.temperature

        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                top_k_values, dim=-1, descending=True
            )
            cumulative_probs = torch.cumsum(
                nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )
            cutoff_idx = torch.searchsorted(cumulative_probs, self.top_p, right=True)
            cutoff_idx = torch.minimum(cutoff_idx + 1, torch.tensor(self.top_k))
            sorted_logits[..., cutoff_idx:] = float("-inf")
            top_k_values.scatter_(-1, sorted_indices, sorted_logits)

        if self.min_p > 0.0:
            probs = nn.functional.softmax(top_k_values, dim=-1)
            highest_prob = torch.max(probs, dim=-1, keepdim=True).values
            min_p = self.min_p * highest_prob
            top_k_values = torch.where(probs < min_p, float("-inf"), top_k_values)

        probs = nn.functional.softmax(top_k_values, dim=-1)
        sample_idx = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(-1, sample_idx)
        return next_token

    @torch.no_grad()
    def generate(self, prompt: str, kv_cache: KVCache):
        prompt_input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # prefill stage
        outputs = self.model(prompt_input, kv_cache=kv_cache)
        next_token = self._sample(outputs[0, -1, :])
        yield next_token.item()

        token_count = 1
        while token_count < self.max_length:
            outputs = self.model(next_token.unsqueeze(0), kv_cache=kv_cache)
            next_token = self._sample(outputs[0, -1, :])

            yield next_token.item()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            token_count += 1

    @torch.no_grad()
    def generate_nonkvcache(self, prompt: str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        generated = input_ids
        token_count = 0

        while token_count < self.max_length:
            outputs = self.model(generated)
            next_token_logits = outputs[0, -1, :]
            next_token = self._sample(next_token_logits)

            yield next_token.item()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=-1)
            token_count += 1

    @torch.no_grad()
    def generate_both(self, prompt: str, kv_cache: KVCache):
        prompt = "Once"
        prompt_input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # prefill stage
        generated = prompt_input
        token_count = 0
        while token_count < self.max_length:
            outputs = self.model(generated[:, -1:], kv_cache=kv_cache)
            # outputs = self.model(generated, kv_cache=kv_cache)
            next_token = self._sample(outputs[0, -1, :])

            yield next_token.item()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=-1)
            token_count += 1
