#!/usr/bin/env python3
"""Measure token usage baseline for Wave 1 gating.

Requires: 50+ representative recall-hook prompts (JSON file with list of dicts)
Output: baseline_tokens.json with per-prompt token counts + summary stats

Usage:
  python scripts/wave1_token_baseline.py --prompts prompts.json --output baseline_tokens.json

Schema (output):
  {
    "schema": "memo.token_baseline.v1",
    "wave": 1,
    "measurements": {
      "prompt_000": 1234,
      "prompt_001": 1567,
      ...
    },
    "total": 123456,
    "mean": 1234.56,
    "median": 1200.0,
    "p95": 2000.0
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def measure_tokens(prompts: list[dict[str, Any]]) -> dict[str, int]:
    """Measure actual token usage for each prompt.

    Each prompt dict should have:
      - "prompt": str (user query)
      - "context": str (optional, recalled memory context)
      - Other fields are ignored for token counting

    Args:
        prompts: List of prompt dicts with "prompt" and optional "context"

    Returns:
        {prompt_id: token_count, ...} where token_count is estimated
        using GPT-4 token approximation (1 token ≈ 4 chars, rough heuristic).
    """
    # Try to import memo's token meter; fallback to simple heuristic if not available
    try:
        from memo.token_meter import estimate_tokens

        logger.info("Using memo.token_meter for token estimation")
        token_fn = estimate_tokens
    except ImportError:
        logger.warning("memo.token_meter not available; using fallback heuristic")

        # Rough heuristic: 1 token ≈ 4 characters (GPT-3/4 approximation)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4 + 1

        token_fn = estimate_tokens

    results: dict[str, int] = {}

    for i, prompt_dict in enumerate(prompts):
        prompt = prompt_dict.get("prompt", "")
        context = prompt_dict.get("context", "")

        # Combine prompt and context for token estimation
        combined_text = prompt + "\n\n" + context if context else prompt
        token_count = token_fn(combined_text)

        prompt_id = f"prompt_{i:03d}"
        results[prompt_id] = token_count

        if (i + 1) % 10 == 0:
            logger.info(f"Measured {i + 1} prompts... ({prompt_id}: {token_count} tokens)")

    return results


def compute_stats(measurements: dict[str, int]) -> dict[str, float | int]:
    """Compute summary statistics from measurements."""
    if not measurements:
        return {"total": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}

    values = list(measurements.values())
    values_sorted = sorted(values)

    total = sum(values)
    mean = total / len(values) if values else 0.0
    median = statistics.median(values) if values else 0.0

    # Compute 95th percentile
    idx_95 = int(0.95 * len(values_sorted))
    p95 = float(values_sorted[idx_95]) if idx_95 < len(values_sorted) else 0.0

    return {
        "total": total,
        "mean": round(mean, 2),
        "median": median,
        "p95": round(p95, 2),
        "min": int(min(values)),
        "max": int(max(values)),
        "count": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Wave 1 token economy baseline for gating decision"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON file with list of prompt dicts (each with 'prompt' and optional 'context')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline_tokens.json"),
        help="Output JSON file for baseline measurements (default: baseline_tokens.json)",
    )
    args = parser.parse_args()

    if not args.prompts.is_file():
        logger.error(f"Prompts file not found: {args.prompts}")
        return 1

    # Load prompts
    try:
        prompts = json.loads(args.prompts.read_text())
        if not isinstance(prompts, list):
            logger.error("Prompts file must contain a JSON array")
            return 1
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return 1

    if not prompts:
        logger.error("Prompts list is empty")
        return 1

    # Measure tokens
    logger.info("Measuring token usage for each prompt...")
    measurements = measure_tokens(prompts)

    # Compute stats
    stats = compute_stats(measurements)

    # Build output
    baseline = {
        "schema": "memo.token_baseline.v1",
        "wave": 1,
        "measurements": measurements,
        **stats,
    }

    # Write output
    args.output.write_text(json.dumps(baseline, indent=2))
    logger.info(f"✓ Baseline saved to {args.output}")
    logger.info(f"  Total tokens: {stats['total']}")
    logger.info(f"  Mean per prompt: {stats['mean']}")
    logger.info(f"  Median per prompt: {stats['median']}")
    logger.info(f"  95th percentile: {stats['p95']}")

    return 0


if __name__ == "__main__":
    exit(main())
