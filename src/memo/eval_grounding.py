"""Ground-truth eval for the grounding detector — does "used memo" match reality?

The utility metric joins recall.log × grounding.log and calls a memory "used"
when used_score ≥ USED_SCORE_STRONG, specific_score ≥ SPECIFIC_MARGIN, or a
downstream action fired. Those thresholds are heuristics; without labels the rate
is unfalsifiable. This module scores the detector against a hand-labeled set:
each label is a `(session_id, turn, recall_id, used: bool)` judgment, and we
report precision / recall / F1 of the detector's decision plus the specific
mistakes, so the bar can be calibrated against truth instead of vibes.

Label schema (`memo.eval_grounding.labels.v1`):
    {"schema": "memo.eval_grounding.labels.v1",
     "labels": [{"session_id": "...", "turn": 7, "recall_id": "ab12cd34",
                 "used": true, "note": "answer cited the decision"}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.dashboard_metrics import grounding_used

LABELS_SCHEMA = "memo.eval_grounding.labels.v1"


@dataclass(frozen=True)
class Label:
    session_id: str
    turn: int
    recall_id: str
    used: bool
    note: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.session_id, self.turn, self.recall_id)


def load_labels(path: Path) -> list[Label]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != LABELS_SCHEMA:
        raise ValueError(f"expected schema {LABELS_SCHEMA}, got {raw.get('schema')!r}")
    out: list[Label] = []
    for row in raw.get("labels", []):
        out.append(
            Label(
                session_id=str(row["session_id"]),
                turn=int(row["turn"]),
                recall_id=str(row["recall_id"])[:8],
                used=bool(row["used"]),
                note=str(row.get("note", "")),
            )
        )
    return out


def detector_used(row: dict[str, Any]) -> bool:
    """The current production "used" decision for one grounding row."""
    return grounding_used(row)


def evaluate(grounding_rows: list[dict[str, Any]], labels: list[Label]) -> dict[str, Any]:
    """Precision/recall/F1 of the detector vs labels, plus the mistakes."""
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for g in grounding_rows:
        sid, turn, rid = g.get("session_id"), g.get("turn"), g.get("recall_id")
        if sid and isinstance(turn, int) and rid:
            by_key[(str(sid), turn, str(rid)[:8])] = g

    tp = fp = fn = tn = 0
    missing = 0
    false_pos: list[Label] = []
    false_neg: list[Label] = []
    for lb in labels:
        row = by_key.get(lb.key)
        if row is None:
            missing += 1
            continue
        pred = detector_used(row)
        if pred and lb.used:
            tp += 1
        elif pred and not lb.used:
            fp += 1
            false_pos.append(lb)
        elif not pred and lb.used:
            fn += 1
            false_neg.append(lb)
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None:
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        f1 = None
    return {
        "labels": len(labels),
        "scored": len(labels) - missing,
        "missing": missing,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "false_positives": [lb.key for lb in false_pos],
        "false_negatives": [lb.key for lb in false_neg],
    }
