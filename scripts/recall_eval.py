#!/usr/bin/env python3
"""Recall-quality eval harness — thin shim over `memo.eval_recall`.

Kept for back-compat / direct invocation. The logic now lives in
`src/memo/eval_recall.py` and is exposed as `memo eval recall`.

    python scripts/recall_eval.py
    python scripts/recall_eval.py --k 3 --json
    python scripts/recall_eval.py --labels mylabels.json --detail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memo import eval_recall
from memo.config import Config
from memo.memory import Memory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--labels", help="JSON label set; defaults to the built-in example")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--detail", action="store_true", help="print per-prompt top-K")
    args = ap.parse_args()

    labels = eval_recall.load_labels(Path(args.labels)) if args.labels else eval_recall.DEFAULT_LABELS
    mem = Memory(Config.from_env())
    rows = eval_recall.evaluate(mem, k=args.k, labels=labels)

    if args.json:
        print(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2))
        return

    print(eval_recall.rows_to_table(rows, args.k))
    print(f"\nRecommendation: {eval_recall.recommend(rows)}")
    if args.detail:
        for r in rows:
            print(f"\n### {r.config}")
            for d in r.detail:
                tag = "scored" if d["scored"] else "probe"
                print(f"  [{tag}] {d['prompt']}")
                for h in d["top"]:
                    flag = "NOISE" if h["noise"] else ("rel" if h["relevant"] else "—")
                    print(f"      {h['score']:>5}  {flag:<5}  {h['title']}")


if __name__ == "__main__":
    main()
