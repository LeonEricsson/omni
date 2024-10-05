import torch
import torch.nn as nn


class Inference:
    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        max_length=50,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        min_p=0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p

        self.model.to(self.device)
        self.model.eval()

    def _sample(self, logits):
        logits = logits / self.temperature

        if self.top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, self.tok_k, dim=-1)
            logits = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_values)

        if self.top_p < 1.0:
            pass

    @torch.no_grad()
    def generate(self, prompt: str):
        input_ids = self.tokenizer(prompt, return_tensor="pt").to(self.device)

        generated = input_ids

        for _ in range(self.max_length):
            outputs = self.model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = self._sample(next_token_logits)
