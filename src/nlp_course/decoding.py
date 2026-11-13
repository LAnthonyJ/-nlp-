"""Single-sequence decoding without a Transformers import or model download."""
import math
import torch


class DecodingSampler:
    def __init__(self, eos_token_id=None, pad_token_id=None):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def apply_temperature(self, logits, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return logits / temperature

    def distribution(self, logits, strategy="greedy", temperature=1.0,
                     top_k=50, top_p=0.9, min_p=0.05):
        if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
            raise ValueError("logits must be a nonempty floating point vector")
        if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
            raise ValueError("logits must contain finite candidates and no NaN/+inf")
        # Softmax is shift-invariant. Center before a small-temperature division
        # so positive logits cannot overflow to +inf (and tied maxima stay zero).
        # Widen arithmetic so half precision and tiny Python floats remain usable.
        working = logits.to(torch.float64)
        if temperature >= 1:
            # Divide first here: subtracting opposite float64 extremes can overflow.
            scaled = self.apply_temperature(working, temperature)
            scaled = scaled - scaled.max()
        else:
            scaled = self.apply_temperature(working - working.max(), temperature)
        probs = torch.softmax(scaled, dim=-1).to(logits.dtype)
        keep = torch.ones_like(probs, dtype=torch.bool)
        if strategy == "greedy":
            keep.zero_()
            keep[probs.argmax()] = True
        elif strategy == "topk":
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError("top_k must be a positive integer")
            keep.zero_()
            keep[probs.topk(min(top_k, probs.numel())).indices] = True
        elif strategy == "topp":
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            values, indices = probs.sort(descending=True)
            # Keep the first token crossing the cumulative probability threshold.
            before = torch.cat((values.new_zeros(1), values.cumsum(0)[:-1]))
            keep.zero_()
            keep[indices[before < top_p]] = True
        elif strategy == "minp":
            if not 0 <= min_p <= 1:
                raise ValueError("min_p must be in [0, 1]")
            keep = probs >= min_p * probs.max()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        probs = probs.masked_fill(~keep, 0)
        return probs / probs.sum()

    def sample(self, logits, generator=None, **kwargs):
        return torch.multinomial(self.distribution(logits, **kwargs), 1, generator=generator).item()

    def topp_step(self, logits, p):
        return self.sample(logits, strategy="topp", top_p=p)

    def minp_step(self, logits, p_min):
        return self.sample(logits, strategy="minp", min_p=p_min)

    @staticmethod
    def penalize_repetition(logits, token_ids, penalty):
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        logits = logits.clone()
        ids = token_ids.unique()
        scores = logits[ids]
        logits[ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
        return logits

    @torch.no_grad()
    def generate(self, model, tokenizer, messages, max_new_tokens=64, strategy="greedy",
                 temperature=1.0, top_k=50, top_p=0.9, min_p=0.05,
                 repetition_penalty=1.0, stop_on_eos=True, device=None, generator=None):
        """Generate one sequence; temporarily disable dropout and restore model mode.

        Uses the full prefix each step (no KV cache); intended for teaching.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        options = dict(strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
        self.distribution(torch.zeros(1), **options)  # Validate even for zero tokens.
        self.penalize_repetition(torch.zeros(1), torch.tensor([0]), repetition_penalty)
        device = device or next(model.parameters()).device
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt")
        generated = enc["input_ids"].to(device)
        if generated.ndim != 2 or generated.shape[0] != 1 or generated.shape[1] == 0:
            raise ValueError("Only a single nonempty prompt is supported")
        attention = enc.get("attention_mask", torch.ones_like(generated)).to(device)
        prefix_length = generated.shape[1]
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            for _ in range(max_new_tokens):
                logits = model(input_ids=generated, attention_mask=attention).logits[0, -1]
                logits = self.penalize_repetition(logits, generated.flatten(), repetition_penalty)
                token = self.sample(logits, generator=generator, **options)
                generated = torch.cat((generated, generated.new_tensor([[token]])), dim=1)
                attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
                if stop_on_eos and token == self.eos_token_id:
                    break
        finally:
            for module, mode in modes:
                module.training = mode
        return tokenizer.decode(generated[0, prefix_length:], skip_special_tokens=True).strip()
