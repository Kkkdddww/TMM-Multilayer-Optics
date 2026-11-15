"""Reflection memory: lessons learned from the realised outcome of past decisions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Lesson:
    ticker: str
    date: str
    action: str
    realized_return: float
    lesson: str

    def render(self) -> str:
        return f"[{self.date} {self.ticker}] action={self.action}, realised return={self.realized_return:+.2%}: {self.lesson}"


class ReflectionMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.items: list[Lesson] = []
        if self.path and self.path.exists():
            self.items = [Lesson(**d) for d in json.loads(self.path.read_text())]

    def add(self, lesson: Lesson) -> None:
        self.items.append(lesson)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(x) for x in self.items], indent=2, ensure_ascii=False))

    def recall(self, ticker: str, before_date: str, k: int = 3) -> list[str]:
        """Most recent lessons strictly before ``before_date``, same ticker first."""
        past = [x for x in self.items if x.date < before_date]
        same = [x for x in past if x.ticker == ticker]
        other = [x for x in past if x.ticker != ticker]
        picked = same[-k:] if k > 0 else []
        if 0 < len(picked) < k:
            picked = other[-(k - len(picked)):] + picked
        elif not picked and k > 0:
            picked = other[-k:]
        return [x.render() for x in picked]
