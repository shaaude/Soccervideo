"""Detector class maps: translate each model's own class numbering onto canonical classes.

Class numbers are NOT universal (guide Step 6). Every map declares the names it expects the
model to report, and `check_model_names` refuses to run when the checkpoint disagrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..schemas import CANONICAL_CLASSES


class ClassMapError(ValueError):
    pass


@dataclass
class ModelClassMap:
    expected_names: dict[int, str]
    map: dict[int, str]
    _lut: dict[int, int] = field(init=False)

    def __post_init__(self):
        self.expected_names = {int(k): str(v) for k, v in (self.expected_names or {}).items()}
        self.map = {int(k): str(v) for k, v in (self.map or {}).items()}
        bad = set(self.map.values()).difference(CANONICAL_CLASSES)
        if bad:
            raise ClassMapError(f"Unknown canonical class(es) {sorted(bad)}; allowed {sorted(CANONICAL_CLASSES)}")
        self._lut = {k: CANONICAL_CLASSES[v] for k, v in self.map.items()}

    def canonical(self, model_class_id: int) -> int | None:
        return self._lut.get(int(model_class_id))

    def check_model_names(self, model_names: dict) -> None:
        names = {int(k): str(v).strip().lower() for k, v in dict(model_names).items()}
        problems = []
        for cid, expected in self.expected_names.items():
            got = names.get(cid)
            if got != expected.strip().lower():
                problems.append(f"class {cid}: expected '{expected}', model reports '{got}'")
        if problems:
            raise ClassMapError("Class map does not match checkpoint: " + "; ".join(problems)
                                + f". Model names: {names}")


@dataclass
class ClassMapConfig:
    name: str
    person_model: ModelClassMap
    ball_model: ModelClassMap | None

    @classmethod
    def load(cls, path) -> "ClassMapConfig":
        raw = yaml.safe_load(Path(path).read_text())
        pm = ModelClassMap(**raw["person_model"])
        bm = ModelClassMap(**raw["ball_model"]) if raw.get("ball_model") else None
        return cls(name=raw.get("name", Path(path).stem), person_model=pm, ball_model=bm)
