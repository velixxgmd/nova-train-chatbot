"""Streaming text generation with chat-template awareness.

Generator.from_checkpoint(run_dir) builds the model from the run's frozen
tokenizer + config and exposes:

  generate(prompt, ...) -> str            (full text)
  stream(prompt, ...)   -> iterator[str]  (chunked, for SSE)
"""
from __future__ import annotations

import json
import os

import torch

from nova.models.model import NovaLM, NovaModelConfig
from nova.tokenization.tokenizer import NovaTokenizer
from nova import chat_template as ct


class Generator:
    def __init__(self, model: NovaLM, tok: NovaTokenizer, device: str = "cpu",
                 repeat_penalty: float = 1.1):
        self.model = model.to(device).eval()
        self.tok = tok
        self.device = device
        self.repeat_penalty = repeat_penalty

    @classmethod
    def from_checkpoint(cls, run_dir: str, checkpoint_name: str = "best",
                        device: str = None, max_seq_len: int = 256):
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        pt = os.path.join(ckpt_dir, f"{checkpoint_name}.pt")
        mj = os.path.join(ckpt_dir, f"{checkpoint_name}.json")
        if not (os.path.exists(pt) and os.path.exists(mj)):
            raise FileNotFoundError(f"checkpoint {checkpoint_name} not found under {ckpt_dir}")
        with open(mj, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tok = NovaTokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
        if meta["vocab_size"] != tok.vocab_size:
            raise RuntimeError(
                f"checkpoint vocab {meta['vocab_size']} != tokenizer vocab {tok.vocab_size}")
        tfp = NovaTokenizer.file_fingerprint(os.path.join(run_dir, "tokenizer.json"))
        if meta.get("tokenizer_fingerprint") and meta["tokenizer_fingerprint"] != tfp:
            raise RuntimeError("tokenizer fingerprint mismatch versus checkpoint")
        cfg = NovaModelConfig.from_dict(meta["model_config"])
        st = torch.load(pt, map_location="cpu", weights_only=False)
        model = NovaLM(cfg)
        model.load_state_dict(st["model"])
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, tok, device)

    # ----------------------------------------------------------------- helpers
    def _sample(self, logits, temperature, top_k, generator, recent_ids):
        logits = logits.float()
        # repetition penalty (simple multiplicative on recent ids)
        if self.repeat_penalty and recent_ids:
            idx = torch.tensor(sorted(set(recent_ids)), dtype=torch.long, device=logits.device)
            logits[idx] /= self.repeat_penalty if temperature > 0 else 1.0
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1))
        logits = logits / max(temperature, 1e-6)
        if top_k:
            k = min(top_k, logits.size(-1))
            v, i = torch.topk(logits, k)
            probs = torch.softmax(v, dim=-1)
            pick = torch.multinomial(probs, 1, generator=generator)
            return int(i[pick])
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))

    # ------------------------------------------------------------------- gen
    @torch.no_grad()
    def stream(self, prompt: str, *, max_new_tokens: int = 64,
               temperature: float = 0.8, top_k: int = 50,
               context_tokens: int = None, seed: int = None):
        """Yield decoded continuation strings token-by-token."""
        context_tokens = context_tokens or self.model.cfg.max_seq_len
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        ids = self.tok.encode(prompt)
        if len(ids) > context_tokens - max_new_tokens:
            ids = ids[-(context_tokens - max_new_tokens):]
        cur = torch.tensor([ids], dtype=torch.long, device=self.device)
        recent = list(ids[-64:])
        recent_texts = []
        for _ in range(max_new_tokens):
            logits, _ = self.model(cur[:, -self.model.cfg.max_seq_len:])
            nxt = self._sample(logits[0, -1], temperature, top_k, g, recent)
            if nxt == self.tok.eos_id:
                break
            recent.append(nxt)
            piece = self.tok.decode([nxt])
            recent_texts.append(piece)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=self.device)], dim=1)
            yield piece

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 50, seed: int = None,
                 stream: bool = False):
        if stream:
            return self.stream(prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_k=top_k, seed=seed)
        out = ""
        for piece in self.stream(prompt, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k, seed=seed):
            out += piece
        return out

    def chat_reply(self, convo: ct.Conversation, *, max_new_tokens: int = 64,
                   temperature: float = 0.8, top_k: int = 50, seed: int = None):
        prompt = ct.render(convo, self.tok, self.model.cfg.max_seq_len - max_new_tokens)
        text = ""
        for piece in self.stream(self.tok.decode(prompt),
                                  max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k, seed=seed):
            text += piece
        return text

    def eval_prompts(self, prompts=("hello", "how are", "the cat", "in a world"),
                     max_new_tokens: int = 48):
        return {p: self.generate(p, max_new_tokens=max_new_tokens, temperature=0.8,
                                 seed=42) for p in prompts}
