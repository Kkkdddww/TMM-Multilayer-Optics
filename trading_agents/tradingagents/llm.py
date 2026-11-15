"""LLM backends.

``ClaudeLLM`` calls the Anthropic API. ``MockLLM`` is a deterministic,
rule-based stand-in so the whole pipeline (and backtests) can run offline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


class LLMRefusal(RuntimeError):
    """The model (and any fallback) declined the request."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, u: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0


class LLM(Protocol):
    usage: Usage

    def text(self, system: str, prompt: str, *, deep: bool = False, hints: dict | None = None) -> str: ...

    def structured(
        self, system: str, prompt: str, schema: dict, *, deep: bool = True, hints: dict | None = None
    ) -> dict: ...


class ClaudeLLM:
    """Claude via the official Anthropic SDK.

    * adaptive thinking (always on for this model), depth controlled by effort
    * quick tasks use ``quick_effort``; decisions use ``deep_effort``
    * structured decisions use JSON-schema structured outputs
    * server-side refusal fallbacks enabled by default
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        quick_effort: str = "low",
        deep_effort: str = "high",
        max_tokens: int = 16000,
        use_server_fallbacks: bool = True,
        client: Any = None,
    ):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.quick_effort, self.deep_effort = quick_effort, deep_effort
        self.max_tokens = max_tokens
        self.use_server_fallbacks = use_server_fallbacks
        self.usage = Usage()

    def _call(self, system: str, prompt: str, effort: str, fmt: dict | None) -> str:
        output_config: dict = {"effort": effort}
        if fmt is not None:
            output_config["format"] = fmt
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config=output_config,
            # Stable system prompt first so repeated calls by the same agent hit the cache.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        if self.use_server_fallbacks:
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            resp = self.client.beta.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        self.usage.add(resp.usage)

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusal(f"model declined: {getattr(details, 'category', None)}")
        if resp.stop_reason == "max_tokens":
            log.warning("response hit max_tokens; output may be truncated")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def text(self, system: str, prompt: str, *, deep: bool = False, hints: dict | None = None) -> str:
        return self._call(system, prompt, self.deep_effort if deep else self.quick_effort, None)

    def structured(
        self, system: str, prompt: str, schema: dict, *, deep: bool = True, hints: dict | None = None
    ) -> dict:
        raw = self._call(
            system,
            prompt,
            self.deep_effort if deep else self.quick_effort,
            {"type": "json_schema", "schema": schema},
        )
        return json.loads(raw)


@dataclass
class MockLLM:
    """Deterministic offline LLM.

    Agents pass a ``hints`` dict with quantitative context (e.g. the technical
    composite score and the agent's role). The mock turns that into plausible
    text and schema-valid decisions, which lets the pipeline, backtester and
    tests run without an API key. It is *not* a trading strategy to rely on.
    """

    threshold: float = 0.15
    usage: Usage = field(default_factory=Usage)
    log: list[dict] = field(default_factory=list)

    def _score(self, hints: dict | None) -> float:
        h = hints or {}
        s = float(h.get("score", 0.0))
        stance = h.get("stance")
        if stance in ("bull", "aggressive"):
            s += 0.2
        elif stance in ("bear", "conservative"):
            s -= 0.2
        return max(-1.0, min(1.0, s))

    def text(self, system: str, prompt: str, *, deep: bool = False, hints: dict | None = None) -> str:
        self.usage.calls += 1
        h = hints or {}
        s = self._score(h)
        role = h.get("role", "agent")
        tone = "bullish" if s > self.threshold else "bearish" if s < -self.threshold else "neutral"
        out = f"[{role}] Overall view: {tone} (signal {s:+.2f})."
        self.log.append({"role": role, "kind": "text", "score": s})
        return out

    def structured(
        self, system: str, prompt: str, schema: dict, *, deep: bool = True, hints: dict | None = None
    ) -> dict:
        self.usage.calls += 1
        s = self._score(hints)
        action = "BUY" if s > self.threshold else "SELL" if s < -self.threshold else "HOLD"
        props = schema.get("properties", {})
        out: dict = {}
        for k, spec in props.items():
            if k == "action":
                out[k] = action
            elif k in ("confidence",):
                out[k] = round(min(1.0, 0.5 + abs(s) / 2), 2)
            elif k == "position_size":
                out[k] = round(min(1.0, abs(s) + 0.25), 2) if action != "HOLD" else 0.5
            elif k == "winner":
                out[k] = "bull" if s > self.threshold else "bear" if s < -self.threshold else "neutral"
            elif spec.get("type") == "number":
                out[k] = 0.0
            elif spec.get("type") == "array":
                out[k] = [f"mock rationale (signal {s:+.2f})"]
            elif "enum" in spec:
                out[k] = spec["enum"][0]
            else:
                out[k] = f"Mock {k}: signal {s:+.2f} -> {action}."
        self.log.append({"role": (hints or {}).get("role"), "kind": "structured", "out": out})
        return out
