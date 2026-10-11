"""Load and enforce the frozen event taxonomy (config/event_taxonomy.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "event_taxonomy.yaml"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    event_types: dict
    review_statuses: tuple
    confirmed_statuses: frozenset

    @property
    def types(self) -> list[str]:
        return list(self.event_types)

    def outcomes(self, event_type: str) -> list[str]:
        return list(self.event_types[event_type]["outcomes"])

    def actor_required(self, event_type: str) -> bool:
        return bool(self.event_types[event_type].get("actor_required", False))

    def machine_only(self, event_type: str) -> bool:
        return bool(self.event_types[event_type].get("machine_only", False))

    def validate_review(self, event_type: str, outcome: str, status: str) -> list[str]:
        errs = []
        if status not in self.review_statuses:
            errs.append(f"unknown review_status '{status}'")
        if event_type not in self.event_types:
            errs.append(f"unknown event_type '{event_type}'")
            return errs
        if outcome not in self.outcomes(event_type):
            errs.append(f"outcome '{outcome}' not valid for {event_type}; allowed {self.outcomes(event_type)}")
        if status in self.confirmed_statuses and self.machine_only(event_type):
            errs.append(f"'{event_type}' is machine-only; choose the observed event type before confirming")
        return errs


@lru_cache(maxsize=8)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
    return Taxonomy(version=raw["version"], event_types=raw["event_types"],
                    review_statuses=tuple(raw["review_statuses"]),
                    confirmed_statuses=frozenset(raw["confirmed_statuses"]))
