"""Chat template and conversation management.

Format (single turn):
    <system> {system} <user> {user} <assistant> {reply}<|eos|>

Handling rules:
  - Loss masking (Stage C): labels for everything except the
    <assistant>…<|eos|> span are -100 so the model only learns to produce
    assistant tokens.
  - Context budget: Conversation.prompt_prompt() truncates the oldest turns
    until the encoded prompt fits max_seq_len minus reserved reply tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Conversation:
    system: str = "You are Nova, a helpful AI assistant."
    turns: list = field(default_factory=list)   # list of {"user": str, "assistant": str}

    def reset(self):
        self.turns.clear()

    def add(self, user: str, assistant: str | None = None):
        self.turns.append({"user": user, "assistant": assistant})


def render(convo: Conversation, tok, max_tokens_for_prompt: int) -> list[int]:
    """Encode the conversation into token ids that fit max_tokens_for_prompt.

    Newest turns are kept; oldest are dropped first. The rendered text always
    ends with <assistant> so the model continues as the assistant.
    """
    sys_ids = tok.encode(f"<system> {convo.system} ")
    budget = max_tokens_for_prompt - len(sys_ids)
    assert budget > 10, "context too small for even the system prompt"

    kept: list = []
    for t in reversed(convo.turns):
        piece = tok.encode(f"<user> {t['user']} ")
        if t["assistant"] is not None:
            piece += tok.encode(f"<assistant> {t['assistant']}{tok.tk.decode([tok.eos_id])} ")
        # tentatively include; roll back if over budget
        if len(kept) + len(piece) > budget:
            break
        kept = piece + kept
    tail = tok.encode("<assistant> ")
    return sys_ids + kept + tail


def build_training_example(convo: Conversation, tok, max_seq_len: int):
    """Build (input_ids, labels) for one dialogue with prompt-token loss masking.

    Assistant tokens (up to and including <|eos|>) get real labels; all other
    positions get -100 (ignored by cross-entropy).
    """
    ids: list = tok.encode(f"<system> {convo.system} ")
    labels: list = [-100] * len(ids)
    for t in convo.turns:
        u = tok.encode(f"<user> {t['user']} ")
        ids += u
        labels += [-100] * len(u)
        if t["assistant"] is not None:
            a = tok.encode(f"<assistant> {t['assistant']}") + [tok.eos_id]
            ids += a
            labels += a[:]          # learn to emit assistant tokens + eos
    if len(ids) >= max_seq_len:
        # keep the final eos as the last labeled token (truncation never drops it)
        ids, labels = ids[: max_seq_len - 1], labels[: max_seq_len - 1]
    ids.append(tok.eos_id)
    labels.append(tok.eos_id)
    return ids, labels
