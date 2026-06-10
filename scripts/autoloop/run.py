#!/usr/bin/env python3
"""Autonomous chat-precision self-debug loop.

Samples random notes, asks Ollama to generate a Spanish question per note,
runs the synapse chat pipeline on those questions, scores via the
label-free signal "seed note id in actual_top_10", buckets failures by
root cause, and twiddles one env-var knob per round.

State persists to ~/.synapse/state/eval/autoloop/. Resumable.

Plan: ~/.claude/plans/valiant-puzzling-blossom.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOME = Path.home()
SYNAPSE_STATE = HOME / ".synapse" / "state"
AUTOLOOP_DIR = SYNAPSE_STATE / "eval" / "autoloop"
CORPUS_PATH = SYNAPSE_STATE / "eval" / "corpus.json"
RUNS_DIR = SYNAPSE_STATE / "eval" / "runs"

OLLAMA_HOST = "http://localhost:11434"
QUESTION_MODEL = "qwen2.5:7b"

QGEN_PROMPT = (
    "Eres generador de preguntas en español rioplatense. "
    "Lee el contenido y emite UNA pregunta natural — la que haría una persona "
    "que necesita ese dato exacto. "
    "Reglas estrictas:\n"
    " - SOLO en español; sin chino ni otros idiomas\n"
    " - SIN palabras 'texto', 'documento', 'nota', 'snippet', 'fragmento', 'contenido'\n"
    " - Sin multi-hop, sin yes/no\n"
    " - Incluí entidades concretas (nombres, fechas, términos técnicos) del cuerpo\n"
    " - Devolvé SOLO la pregunta, sin prefacio ni explicación\n\n"
    "TÍTULO: {title}\n"
    "CUERPO:\n{body}\n"
)

KNOB_LADDERS: dict[str, list[tuple[str, str]]] = {
    "R-miss": [
        ("MEMO_RERANK_INPUT_K", "50"),
        ("MEMO_RERANK_INPUT_K", "80"),
        ("SYNAPSE_HYDE", "1"),
        ("SYNAPSE_HYDE_N", "3"),
        ("SYNAPSE_MULTI_QUERY", "1"),
        ("SYNAPSE_MULTI_QUERY_N", "3"),
    ],
    "R-rank": [
        ("MEMO_RERANK_FUSION_ALPHA", "0.85"),
        ("SYNAPSE_RERANK", "local"),
        ("SYNAPSE_RERANK", "llm"),
    ],
    "S-miss": [
        ("SYNAPSE_MULTI_SOURCE_SYNTHESIS", "1"),
        ("SYNAPSE_MULTI_SOURCE_TOP_N", "5"),
        ("SYNAPSE_MULTI_SOURCE_MIN_DISTINCT", "2"),
    ],
}

ALWAYS_ON_ENV = {"SYNAPSE_STATE_DIR": str(SYNAPSE_STATE)}


# ----------------------------- utilities ------------------------------ #


def now_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, timeout: float = 900.0) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=timeout, check=False)


# ----------------------------- ollama --------------------------------- #


def ollama_start() -> None:
    log("ollama start")
    subprocess.run(["brew", "services", "start", "ollama"], check=False, capture_output=True)
    # wait until /api/version responds
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ollama did not come up within 60s")


def ollama_stop() -> None:
    log("ollama stop")
    subprocess.run(["brew", "services", "stop", "ollama"], check=False, capture_output=True)
    subprocess.run(["pkill", "-f", "ollama runner"], check=False, capture_output=True)
    time.sleep(2)


def ollama_generate(prompt: str, model: str = QUESTION_MODEL, temperature: float = 0.3) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": 80}})
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=body.encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return str(payload.get("response", "")).strip()


# ----------------------------- memo + corpus ------------------------- #


def memo_list_ids(seen: set[str]) -> list[str]:
    res = run_cmd(["memo", "list", "--limit", "1200", "--json"])
    if res.returncode != 0:
        raise RuntimeError(f"memo list failed: {res.stderr[:500]}")
    rows = json.loads(res.stdout)
    eligible = [r for r in rows if r.get("id") and r["id"] not in seen and len(r.get("body") or "") >= 200]
    return [r["id"] for r in eligible]


def memo_get(nid: str) -> dict:
    res = run_cmd(["memo", "get", nid, "--json"])
    if res.returncode != 0:
        raise RuntimeError(f"memo get {nid} failed: {res.stderr[:300]}")
    return json.loads(res.stdout)


_token_re = re.compile(r"\w+", re.UNICODE)
_ES_STOP = frozenset({"a","al","de","del","el","en","es","la","las","lo","los","mi","mis","no","por","que","qué",
                      "se","si","sí","su","sus","un","una","y","ya","tu","tus","con","sin","sobre","para",
                      "cuando","cuándo","donde","dónde","como","cómo","cual","cuál"})


def content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _token_re.findall(text) if t.lower() not in _ES_STOP and len(t) >= 3}


_REFERS_TO_DOC = re.compile(r"\b(texto|documento|nota|snippet|fragmento|contenido)\b", re.IGNORECASE)
_NON_LATIN = re.compile(r"[　-鿿一-鿿]")  # CJK ideographs


def validate_question(q: str, body: str) -> bool:
    if not (15 <= len(q) <= 200):
        return False
    if q.count("?") != 1:
        return False
    if _REFERS_TO_DOC.search(q):
        return False
    if _NON_LATIN.search(q):
        return False
    overlap = content_tokens(q) & content_tokens(body)
    return len(overlap) >= 2


def generate_question(note: dict) -> str | None:
    title = note.get("title") or ""
    body = (note.get("body") or "")[:1800]
    prompt = QGEN_PROMPT.format(title=title, body=body)
    for attempt in range(3):
        try:
            q = ollama_generate(prompt, temperature=0.3 + 0.1 * attempt)
        except Exception as exc:
            log(f"  qgen error: {exc!r}")
            return None
        # strip wrapping quotes / leading dashes the model sometimes adds
        q = q.strip().strip('"\'').strip()
        if "\n" in q:
            q = q.split("\n", 1)[0].strip()
        if validate_question(q, body):
            return q
        log(f"  qgen reject [{attempt}]: {q[:100]!r}")
    return None


def write_corpus(pairs: list[tuple[str, str]]) -> None:
    rows = [
        {
            "id": f"auto-{nid[:12]}",
            "question": q,
            "expected_source_ids": [f"memo:{nid}"],
            "notes": "autoloop seed",
            "source": "autoloop",
            "chat_session_id": "",
            "labeled_at": datetime.now(UTC).isoformat(),
            "rating": "",
            "auto_labeled": True,
            "needs_review": False,
            "schema": "synapse.eval_chat.query.v1",
        }
        for nid, q in pairs
    ]
    save_json(CORPUS_PATH, rows)


def backup_corpus_once(autoloop_dir: Path) -> None:
    marker = autoloop_dir / ".corpus_backed_up"
    if marker.exists() or not CORPUS_PATH.exists():
        return
    autoloop_dir.mkdir(parents=True, exist_ok=True)
    backup = autoloop_dir / f"corpus.bak.{now_ts()}.json"
    shutil.copy2(CORPUS_PATH, backup)
    marker.write_text(str(backup), encoding="utf-8")
    log(f"corpus backed up to {backup}")


# ----------------------------- eval ----------------------------------- #


def run_synapse_eval(env: dict[str, str]) -> Path:
    full = {**ALWAYS_ON_ENV, **env}
    log(f"  synapse eval-chat run --no-judge (env diff: { {k:v for k,v in env.items() if k not in ALWAYS_ON_ENV} })")
    res = run_cmd(["synapse", "eval-chat", "run", "--no-judge", "--json"], env=full, timeout=1800.0)
    if res.returncode != 0:
        log(f"  eval-chat stderr: {res.stderr[-500:]}")
        raise RuntimeError("synapse eval-chat failed")
    # newest run file
    runs = sorted(RUNS_DIR.glob("*.json"))
    if not runs:
        raise RuntimeError("no run file emitted")
    return runs[-1]


def memo_id_part(source_id: str) -> str:
    raw = str(source_id or "").strip().lower()
    if raw.startswith("memo://memoria/"):
        return raw.rsplit("/", 1)[-1]
    if raw.startswith("memo:"):
        return raw.split(":", 1)[1]
    if re.fullmatch(r"[0-9a-f]{8,64}", raw):
        return raw
    return ""


def ids_match(actual_id: str, expected_id: str) -> bool:
    actual = str(actual_id or "").strip().lower()
    expected = str(expected_id or "").strip().lower()
    if actual == expected:
        return True
    actual_memo = memo_id_part(actual)
    expected_memo = memo_id_part(expected)
    if actual_memo and expected_memo:
        shorter = min(len(actual_memo), len(expected_memo))
        return shorter >= 8 and (
            actual_memo.startswith(expected_memo)
            or expected_memo.startswith(actual_memo)
        )
    return False


def score_run(run_path: Path, seed_pairs: list[tuple[str, str]]) -> list[dict]:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    by_q = {r["question"]: r for r in data.get("per_query", [])}
    out: list[dict] = []
    for q, nid in [(q, nid) for nid, q in seed_pairs]:
        r = by_q.get(q)
        if r is None:
            out.append({"nid": nid, "question": q, "missing": True})
            continue
        actual = [str(x).lower() for x in (r.get("actual_top_10") or [])]
        target = f"memo:{nid}".lower()
        rank = next(
            (
                idx
                for idx, actual_id in enumerate(actual, start=1)
                if ids_match(actual_id, target)
            ),
            11,
        )
        retrieval_hit = rank <= 10
        top1_hit = rank == 1
        answer = r.get("answer", "")
        nid_short = nid[:8]
        cited = nid_short in answer or any(nid_short in (s.get("id") or "") for s in (r.get("sources") or []))
        out.append({
            "nid": nid,
            "question": q,
            "rank": rank,
            "retrieval_hit": retrieval_hit,
            "top1_hit": top1_hit,
            "cited": cited,
            "answer_head": answer[:200],
            "latency_ms": r.get("total_ms", 0),
            "missing": False,
        })
    return out


def bucket_failure(row: dict) -> str | None:
    if row.get("missing"):
        return None
    if not row["retrieval_hit"]:
        return "R-miss"
    if row["rank"] > 2:
        return "R-rank"
    if not row["cited"]:
        return "S-miss"
    return None


def aggregate_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("missing")]
    n = len(valid) or 1
    recall_at_10 = sum(1 for r in valid if r["retrieval_hit"]) / n
    mrr = sum(1.0 / r["rank"] if r["rank"] <= 10 else 0.0 for r in valid) / n
    cite_rate = sum(1 for r in valid if r["cited"]) / n
    mean_rank = sum(r["rank"] for r in valid) / n
    return {
        "n": n,
        "recall_at_10": round(recall_at_10, 4),
        "mrr": round(mrr, 4),
        "cite_rate": round(cite_rate, 4),
        "mean_rank": round(mean_rank, 2),
    }


# ----------------------------- tuning --------------------------------- #


def pick_dominant_bucket(buckets: list[str]) -> str | None:
    if not buckets:
        return None
    counts: dict[str, int] = {}
    for b in buckets:
        counts[b] = counts.get(b, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def next_knob(bucket: str, tried: dict[str, list[int]]) -> tuple[str, str] | None:
    ladder = KNOB_LADDERS.get(bucket, [])
    used = set(tried.get(bucket, []))
    for i, kv in enumerate(ladder):
        if i not in used:
            return kv, i  # type: ignore[return-value]
    return None


# ----------------------------- state ---------------------------------- #


@dataclass
class LoopState:
    round_no: int = 0
    knob_state: dict[str, str] = field(default_factory=dict)
    ladder_state: dict[str, list[int]] = field(default_factory=dict)
    seen_ids: set[str] = field(default_factory=set)
    consec_plateau: int = 0
    consec_success: int = 0
    prev_recall: float = 0.0
    history: list[dict] = field(default_factory=list)


def load_state() -> LoopState:
    s = LoopState()
    s.knob_state = load_json(AUTOLOOP_DIR / "knob_state.json", {})
    s.ladder_state = load_json(AUTOLOOP_DIR / "ladder_state.json", {})
    s.seen_ids = set(load_json(AUTOLOOP_DIR / "seen_ids.json", []))
    metas = sorted(AUTOLOOP_DIR.glob("round-*.meta.json"))
    if metas:
        last = json.loads(metas[-1].read_text(encoding="utf-8"))
        s.round_no = int(last.get("round", 0))
        s.prev_recall = float(last.get("metrics", {}).get("recall_at_10", 0.0))
        s.consec_plateau = int(last.get("consec_plateau", 0))
        s.consec_success = int(last.get("consec_success", 0))
        s.history = [json.loads(p.read_text(encoding="utf-8")) for p in metas]
    return s


def persist_state(s: LoopState) -> None:
    save_json(AUTOLOOP_DIR / "knob_state.json", s.knob_state)
    save_json(AUTOLOOP_DIR / "ladder_state.json", s.ladder_state)
    save_json(AUTOLOOP_DIR / "seen_ids.json", sorted(s.seen_ids))


# ----------------------------- summary -------------------------------- #


def write_summary(s: LoopState, reason: str) -> None:
    lines = [
        "# Autoloop summary",
        "",
        f"Stopped: **{reason}**",
        f"Rounds run: {s.round_no}",
        f"Final Recall@10: {s.prev_recall:.4f}",
        "",
        "## Final knob state",
        "",
        "```",
    ]
    if s.knob_state:
        for k in sorted(s.knob_state):
            lines.append(f"export {k}={s.knob_state[k]}")
    else:
        lines.append("(no overrides — defaults best)")
    lines.append("```")
    lines.append("")
    lines.append("## Per-round table")
    lines.append("")
    lines.append("| round | recall@10 | mrr | cite | mean_rank | knob_diff |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for h in s.history:
        m = h.get("metrics", {})
        kd = h.get("knob_diff", {}) or {}
        diff = ", ".join(f"{k}={v}" for k, v in kd.items()) or "—"
        lines.append(
            f"| {h.get('round')} | {m.get('recall_at_10',0):.3f} | {m.get('mrr',0):.3f} | "
            f"{m.get('cite_rate',0):.3f} | {m.get('mean_rank',0):.2f} | {diff} |"
        )
    (AUTOLOOP_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"summary written to {AUTOLOOP_DIR / 'summary.md'}")


# ----------------------------- main loop ------------------------------ #


def one_round(s: LoopState, n_per_round: int, no_tune: bool) -> dict:
    s.round_no += 1
    log(f"=== round {s.round_no} ===")

    # 1. Sample
    eligible = memo_list_ids(s.seen_ids)
    if len(eligible) < n_per_round:
        log("  pool exhausted; resetting seen_ids")
        s.seen_ids.clear()
        eligible = memo_list_ids(s.seen_ids)
    sample_ids = random.sample(eligible, min(n_per_round, len(eligible)))
    log(f"  sampled {len(sample_ids)} ids")

    # 2. Generate questions via Ollama
    ollama_start()
    try:
        pairs: list[tuple[str, str]] = []
        for nid in sample_ids:
            note = memo_get(nid)
            q = generate_question(note)
            if q:
                pairs.append((nid, q))
                log(f"  Q[{nid[:8]}]: {q[:100]}")
            else:
                log(f"  Q[{nid[:8]}]: REJECTED")
    finally:
        ollama_stop()

    if not pairs:
        log("  no valid questions — skipping eval")
        return {"round": s.round_no, "metrics": {"recall_at_10": s.prev_recall, "mrr": 0, "cite_rate": 0, "mean_rank": 0, "n": 0}, "buckets": {}, "knob_diff": {}}

    s.seen_ids.update(nid for nid, _ in pairs)

    # 3. Write corpus
    write_corpus(pairs)
    log(f"  wrote corpus with {len(pairs)} queries")

    # 4. Run eval
    run_path = run_synapse_eval(s.knob_state)

    # 5. Score
    rows = score_run(run_path, pairs)
    metrics = aggregate_metrics(rows)
    log(f"  metrics: {metrics}")

    # 6. Bucket failures
    buckets = [b for b in (bucket_failure(r) for r in rows) if b]
    bucket_hist = {b: buckets.count(b) for b in set(buckets)}
    log(f"  buckets: {bucket_hist}")

    # 7. Tune (one knob)
    knob_diff: dict[str, str] = {}
    if not no_tune:
        dom = pick_dominant_bucket(buckets)
        if dom and len(buckets) >= max(1, metrics["n"] // 3):
            nxt = next_knob(dom, s.ladder_state)
            if nxt is not None:
                (k, v), idx = nxt
                s.knob_state[k] = v
                s.ladder_state.setdefault(dom, []).append(idx)
                knob_diff = {k: v}
                log(f"  tune: {dom} → {k}={v} (ladder idx {idx})")

    # 8. Plateau/success tracking
    delta = metrics["recall_at_10"] - s.prev_recall
    if metrics["recall_at_10"] >= 0.85:
        s.consec_success += 1
    else:
        s.consec_success = 0
    all_exhausted = all(
        len(s.ladder_state.get(b, [])) >= len(KNOB_LADDERS[b]) for b in KNOB_LADDERS
    )
    if abs(delta) < 0.03 and all_exhausted:
        s.consec_plateau += 1
    else:
        s.consec_plateau = 0
    s.prev_recall = metrics["recall_at_10"]

    # 9. Persist
    append_jsonl(AUTOLOOP_DIR / f"round-{s.round_no:03d}.jsonl",
                 {"round": s.round_no, "metrics": metrics, "rows": rows})
    meta = {
        "round": s.round_no,
        "knob_state": dict(s.knob_state),
        "knob_diff": knob_diff,
        "metrics": metrics,
        "bucket_histogram": bucket_hist,
        "seed_ids": [nid for nid, _ in pairs],
        "run_file": str(run_path),
        "consec_plateau": s.consec_plateau,
        "consec_success": s.consec_success,
    }
    save_json(AUTOLOOP_DIR / f"round-{s.round_no:03d}.meta.json", meta)
    s.history.append(meta)
    persist_state(s)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--hard-cap-hours", type=float, default=6.0)
    ap.add_argument("--n-per-round", type=int, default=5)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    AUTOLOOP_DIR.mkdir(parents=True, exist_ok=True)
    backup_corpus_once(AUTOLOOP_DIR)

    s = load_state()
    start_round = s.round_no
    t0 = time.time()
    reason = "unknown"

    def _signal_exit(signum, frame):
        nonlocal reason
        reason = f"signal-{signum}"
        write_summary(s, reason)
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_exit)
    signal.signal(signal.SIGTERM, _signal_exit)

    try:
        while True:
            if s.round_no - start_round >= args.max_rounds:
                reason = f"max-rounds ({args.max_rounds})"
                break
            if (time.time() - t0) / 3600.0 >= args.hard_cap_hours:
                reason = f"hard-cap-hours ({args.hard_cap_hours})"
                break
            if s.consec_success >= 2:
                reason = "success (Recall@10>=0.85 x2)"
                break
            if s.consec_plateau >= 3:
                reason = "plateau (3 rounds, ladders exhausted)"
                break
            try:
                one_round(s, args.n_per_round, args.no_tune)
            except Exception as exc:
                log(f"round error: {exc!r}")
                # do not crash the loop on a single round failure
                time.sleep(5)
    finally:
        write_summary(s, reason)

    log(f"done: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
