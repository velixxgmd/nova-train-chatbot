"""Basic safety boundaries.

Honest scope: this is a keyword-layer guard, not a classifier. It refuses
obvious harmful requests by returning a canned refused message instead of
model output. Tool execution is NOT supported anywhere in v1 regardless.
"""
from __future__ import annotations

import re

REFUSAL = ("I'm sorry, but I can't help with that.")

_PATTERNS = [
    r"\b(how to (build|make|synthesize|buy)|instructions for (building|making))\s+(a\s+)?(bomb|explosive|bioweapon|nerve agent|meth|ricin)",
    r"\b(kill|murder|assassinate)\s+(a\s+)?(specific\s+)?(person|my (boss|spouse|neighbor)|stepfather|mother|father)",
    r"\b(child|csam|porn)\b.*\b(abus|exploit|generate)",
    r"\b(remote\s+access\s+trojan|keylogger|ransomware|ddos\s+tool|stolen (credit card|credentials))",
    r"\b(self[- ]harm|suicide)\b",
]


def is_prohibited(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _PATTERNS)


def safe_completion_or_refusal(prompt: str, run_generation) -> str:
    """run_generation is a zero-arg callable returning generated text."""
    if is_prohibited(prompt):
        return REFUSAL
    return run_generation()
