#!/usr/bin/env python3
"""Recall-quality eval harness (Fase 0).

Measures recall PRECISION against the live memo corpus under several configs,
so the default recall mode/floor is chosen from data, not by eye.

For each labeled prompt we run ``Memory.search`` under a config, apply the same
post-filters the recall-hook uses (similarity floor, optional archive
exclusion), take the top-K, and classify each hit as relevant / noise / neutral
by tags+path heuristics. Reports precision@K and noise@K per config.

Run with an interpreter that can import ``memo`` (the installed memo, or
``PYTHONPATH=src`` with deps):

    python scripts/recall_eval.py
    python scripts/recall_eval.py --k 3 --json

Configs:
  A  vec    floor 0.60  no archive exclusion   (current default / baseline)
  B  vec    floor 0.72  archive-excluded
  C  hybrid floor 0.40  archive-excluded       (cross-encoder rerank if enabled)
  D  hybrid floor 0.40  archive-excluded + session-context blend
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

from memo.config import Config
from memo.memory import Memory

# --- Labeled prompt set ------------------------------------------------------
# `stack` prompts SHOULD surface the 3-pillar / memo-internals memories and
# should NOT surface archived HR/finance notes. `generic` prompts are off-topic
# probes: archived notes must NOT masquerade as relevant.

STACK_PROMPTS = [
    "cómo está la arquitectura del stack synapse memflow memo",
    "qué falta para retomar Tier 2 del stack de consciencia",
    "dónde está el registro de puertos y daemons del stack",
    "cómo funciona el recall de memo y por qué trae ruido",
    "qué decidí sobre el gate de aliases memory_* en memflow",
]
GENERIC_PROMPTS = [
    "qué reunión tengo agendada para mañana",
    "receta para hacer una tarta de manzana",
]

# Tokens that mark a hit as on-topic for the stack prompts.
STACK_TERMS = {
    "synapse", "memflow", "memo", "3-pilares", "stack", "tier-2", "tier 2",
    "arquitectura", "recall", "provenance", "consciousness", "consciencia",
    "onboarding", "daemon", "mcp",
}
# Tokens / path fragments that mark a hit as archived/off-topic NOISE.
NOISE_TAGS = {
    "04-archive", "old", "moka", "foda", "swot", "aws", "aws-tagging",
    "hr", "1a1", "companies",
}
NOISE_PATH_FRAGMENTS = ("inactive/", "/04-archive/", "04-archive/", "/old/", "/companies/")

# Session-context blend used by config D (simulates prompt_trail / running_summary).
SESSION_CONTEXT = (
    "trabajando en el stack de consciencia: synapse, memflow, memo; "
    "git, recall, provenance, tier 2"
)


def _is_noise(rec) -> bool:
    tags = {str(t).lower() for t in (rec.tags or [])}
    if tags & NOISE_TAGS:
        return True
    p = (rec.path or "").lower()
    return any(frag in p for frag in NOISE_PATH_FRAGMENTS)


def _is_stack_relevant(rec) -> bool:
    if _is_noise(rec):
        return False
    hay = " ".join([
        (rec.title or ""), " ".join(rec.tags or []), (rec.path or ""),
        (rec.body or "")[:200],
    ]).lower()
    return any(term in hay for term in STACK_TERMS)


@dataclass
class Cfg:
    name: str
    mode: str
    floor: float
    exclude_archived: bool
    context: bool = False


CONFIGS = [
    Cfg("A vec/0.60/keep", "vec", 0.60, exclude_archived=False),
    Cfg("B vec/0.72/excl", "vec", 0.72, exclude_archived=True),
    Cfg("C hyb/0.40/excl", "hybrid", 0.40, exclude_archived=True),
    Cfg("D hyb/0.40/ctx", "hybrid", 0.40, exclude_archived=True, context=True),
]


@dataclass
class Row:
    config: str
    precision_at_k: float = 0.0
    noise_at_k: float = 0.0
    latency_ms_p50: float = 0.0
    detail: list = field(default_factory=list)


def run_config(mem: Memory, cfg: Cfg, k: int) -> Row:
    lat: list[float] = []
    prec_hits = 0
    prec_total = 0
    noise_hits = 0
    detail = []
    for prompt in STACK_PROMPTS + GENERIC_PROMPTS:
        is_stack = prompt in STACK_PROMPTS
        query = f"{SESSION_CONTEXT}\n{prompt}" if cfg.context else prompt
        t0 = time.time()
        hits = mem.search(query, limit=k * 4, mode=cfg.mode)
        lat.append((time.time() - t0) * 1000)
        # floor
        hits = [h for h in hits if h.score is None or h.score >= cfg.floor]
        # archive exclusion
        if cfg.exclude_archived:
            hits = [h for h in hits if not _is_noise(h)]
        top = hits[:k]
        n_noise = sum(1 for h in top if _is_noise(h))
        noise_hits += n_noise
        if is_stack:
            prec_total += k
            prec_hits += sum(1 for h in top if _is_stack_relevant(h))
        detail.append({
            "prompt": prompt[:48],
            "stack": is_stack,
            "top": [
                {"title": (h.title or "")[:40], "score": round(h.score or 0, 3),
                 "noise": _is_noise(h), "relevant": _is_stack_relevant(h)}
                for h in top
            ],
        })
    lat.sort()
    return Row(
        config=cfg.name,
        precision_at_k=round(prec_hits / prec_total, 3) if prec_total else 0.0,
        noise_at_k=round(noise_hits / (len(STACK_PROMPTS + GENERIC_PROMPTS) * k), 3),
        latency_ms_p50=round(lat[len(lat) // 2], 1) if lat else 0.0,
        detail=detail,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--detail", action="store_true", help="print per-prompt top-K")
    args = ap.parse_args()

    mem = Memory(Config.from_env())
    rows = [run_config(mem, cfg, args.k) for cfg in CONFIGS]

    if args.json:
        print(json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2))
        return

    print(f"\nRecall eval — precision@{args.k} (stack prompts) / noise@{args.k} (all)\n")
    print(f"{'config':<18} {'prec@k':>7} {'noise@k':>8} {'p50 ms':>8}")
    print("-" * 45)
    for r in rows:
        print(f"{r.config:<18} {r.precision_at_k:>7} {r.noise_at_k:>8} {r.latency_ms_p50:>8}")
    print("\nHigher prec@k + lower noise@k is better. Baseline = config A.")
    if args.detail:
        for r in rows:
            print(f"\n### {r.config}")
            for d in r.detail:
                tag = "STACK" if d["stack"] else "generic"
                print(f"  [{tag}] {d['prompt']}")
                for h in d["top"]:
                    flag = "NOISE" if h["noise"] else ("rel" if h["relevant"] else "—")
                    print(f"      {h['score']:>5}  {flag:<5}  {h['title']}")


if __name__ == "__main__":
    main()
