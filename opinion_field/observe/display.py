"""§4.4 display-name mapping — applied ONLY when rendering. Never imported by the engine."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import load_yaml
from ..schema import SLOTS


class DisplayMap:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = mapping or {}

    @classmethod
    def load(cls, path: str | Path | None) -> "DisplayMap":
        if not path or not Path(path).exists():
            return cls()
        raw = load_yaml(path).get("slots", {})
        return cls({k: str(v.get("display", SLOTS.get(k, k))) for k, v in raw.items()})

    def render(self, text: str) -> str:
        """Replace slot labels (후보 A, 정당 갑, 이슈 I …) with display names."""
        if not self.mapping:
            return text
        for slot, label in SLOTS.items():
            disp = self.mapping.get(slot)
            if disp and disp != label:
                text = re.sub(re.escape(label), disp, text)
        return text

    def name(self, slot: str) -> str:
        return self.mapping.get(slot, SLOTS.get(slot, slot))
