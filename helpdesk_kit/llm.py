"""Talking to Claude: the client, model names, a chat helper and cost."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from getpass import getpass
from threading import Lock
from typing import Any

import anthropic

SMALL = "claude-haiku-5-5"
MID = "claude-sonnet-5-5"
LARGE = "claude-opus-5-5"

# Every notebook uses this unless it says otherwise. Change it here, once.
DEFAULT_MODEL = MID

# US dollars per million tokens. While a price is None, cost() returns None
# and the notebooks print "n/a". Tokens are always reported.
PRICES = {
    SMALL: {"input": None, "output": None},
    MID: {"input": None, "output": None},
    LARGE: {"input": None, "output": None},
}


def _read_key() -> str:
    """Environment first (headless test runs), then Colab secrets, then a prompt."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata

        key = userdata.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:  # Not on Colab, secret missing, or access not granted.
        pass
    return getpass("Paste your Anthropic API key: ")


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Return one shared client. The key is never printed."""
    return anthropic.Anthropic(api_key=_read_key(), max_retries=4)


def cost(usage: Any, model: str) -> float | None:
    """Cost of one call in dollars, or None if the model has no price yet."""
    price = PRICES.get(model)
    if not price or price["input"] is None or price["output"] is None:
        return None
    return (
        usage.input_tokens * price["input"] + usage.output_tokens * price["output"]
    ) / 1_000_000


def format_cost(dollars: float | None) -> str:
    return "n/a" if dollars is None else f"${dollars:.5f}"


@dataclass
class Reply:
    """What chat() returns: the text, plus what it took to get it."""

    text: str
    thinking: str
    model: str
    usage: Any
    seconds: float
    stop_reason: str | None
    raw: Any

    @property
    def cost(self) -> float | None:
        return cost(self.usage, self.model)

    def stats(self) -> str:
        return (
            f"{self.model} · {self.usage.input_tokens} in / {self.usage.output_tokens} out"
            f" · {format_cost(self.cost)} · {self.seconds:.1f}s"
        )


def chat(
    prompt: str | list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> Reply:
    """Send one request. `prompt` is a string or a full list of messages.

    Extra keyword arguments (thinking, tools, ...) go straight to the API.
    """
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    params = {"model": model, "max_tokens": max_tokens, "messages": messages, **kwargs}
    if system is not None:
        params["system"] = system

    start = time.perf_counter()
    response = get_client().messages.create(**params)
    seconds = time.perf_counter() - start

    reply = Reply(
        text="".join(block.text for block in response.content if block.type == "text"),
        thinking="".join(
            block.thinking for block in response.content if block.type == "thinking"
        ),
        model=model,
        usage=response.usage,
        seconds=seconds,
        stop_reason=response.stop_reason,
        raw=response,
    )
    _log_usage(reply)
    return reply


class Meter:
    """Adds up tokens and cost across calls. Safe to share between threads."""

    def __init__(self):
        self._lock = Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._costs: list[float | None] = []

    def add(self, reply: Reply) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += reply.usage.input_tokens
            self.output_tokens += reply.usage.output_tokens
            self._costs.append(reply.cost)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float | None:
        """None if any call was unpriced: the total is unknown, not lower."""
        return None if None in self._costs or not self._costs else sum(self._costs)


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Tokens in one user message, including a small fixed overhead."""
    result = get_client().messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    return result.input_tokens


def extract_json(text: str) -> str:
    """Cut a reply down to its outermost {...}, dropping code fences and chatter."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def parse_json(text: str) -> dict:
    return json.loads(extract_json(text))


def _log_usage(reply: Reply) -> None:
    """scripts/run_all.py sets this variable to total tokens and cost per notebook."""
    path = os.environ.get("HELPDESK_USAGE_LOG")
    if not path:
        return
    record = {
        "model": reply.model,
        "input_tokens": reply.usage.input_tokens,
        "output_tokens": reply.usage.output_tokens,
        "cost": reply.cost,
    }
    with open(path, "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
