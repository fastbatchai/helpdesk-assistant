"""Tracing: OpenTelemetry spans, plus a local copy to analyse.

Without a backend the spans go nowhere, but SPANS still fills up. Call
setup_langfuse() or setup_phoenix() to send them to a UI as well.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import pandas as pd
from opentelemetry import trace

SPANS: list[dict] = []
_PRIMITIVES = (str, int, float, bool)
# Phoenix and Langfuse show these two attributes as a step's Input and Output.
_INPUT_OUTPUT = {"question": "input.value", "prompt": "input.value", "query": "input.value",
                 "answer": "output.value", "completion": "output.value", "chunks": "output.value"}


def _secret(name: str) -> str | None:
    """Read a value from the environment, then Colab secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:  # Not on Colab, or the secret isn't set.
        return None


def setup_langfuse() -> bool:
    """Send spans to Langfuse cloud. Returns False if the keys are missing."""
    public, secret = _secret("LANGFUSE_PUBLIC_KEY"), _secret("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return False
    os.environ["LANGFUSE_PUBLIC_KEY"] = public
    os.environ["LANGFUSE_SECRET_KEY"] = secret
    os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")

    from langfuse import get_client

    return get_client().auth_check()


def setup_phoenix(project: str = "helpdesk") -> str:
    """Start Phoenix inside the runtime. No account needed. Returns its URL."""
    import phoenix
    from phoenix.otel import register

    session = phoenix.launch_app()
    register(project_name=project, batch=False)
    return session.url


def flush() -> None:
    """Push any waiting spans to the backend."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


class _Span:
    def __init__(self, otel_span, record: dict):
        self._otel = otel_span
        self.record = record

    def set(self, **attributes) -> None:
        self.record.update(attributes)
        for key, value in attributes.items():
            value = value if isinstance(value, _PRIMITIVES) else str(value)
            self._otel.set_attribute(key, value)
            if key in _INPUT_OUTPUT:
                self._otel.set_attribute(_INPUT_OUTPUT[key], value)

    def add_usage(self, reply) -> None:
        """Attach model, tokens and cost, using the OpenTelemetry GenAI attribute names."""
        self.record.update(
            model=reply.model,
            input_tokens=self.record["input_tokens"] + reply.usage.input_tokens,
            output_tokens=self.record["output_tokens"] + reply.usage.output_tokens,
            cost=reply.cost,
        )
        self._otel.set_attribute("gen_ai.request.model", reply.model)
        self._otel.set_attribute("gen_ai.usage.input_tokens", reply.usage.input_tokens)
        self._otel.set_attribute("gen_ai.usage.output_tokens", reply.usage.output_tokens)


@contextmanager
def span(name: str, **attributes):
    """Time a block of work and record it in SPANS and in any tracing backend."""
    record = {"name": name, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "cost": None}
    with trace.get_tracer("helpdesk_kit").start_as_current_span(name) as otel_span:
        wrapped = _Span(otel_span, record)
        wrapped.set(**attributes)
        start = time.perf_counter()
        try:
            yield wrapped
        finally:
            record["seconds"] = round(time.perf_counter() - start, 3)
            SPANS.append(record)


def span_table() -> pd.DataFrame:
    """Time and tokens per kind of step, slowest first."""
    frame = pd.DataFrame(SPANS)
    if frame.empty:
        return frame
    frame["tokens"] = frame["input_tokens"] + frame["output_tokens"]
    table = frame.groupby("name").agg(
        calls=("seconds", "size"),
        total_seconds=("seconds", "sum"),
        avg_seconds=("seconds", "mean"),
        total_tokens=("tokens", "sum"),
    )
    return table.sort_values("total_seconds", ascending=False).round(3)
