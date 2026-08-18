"""Dream observability passes — label harvest + nightly retrieval eval.

Covers `_run_harvest_labels` / `_run_eval_recall` (cli_dream_passes) and their
wiring in `memo dream run` (MEMO_DREAM_EVAL_ENABLED gate, receipt fragments,
history.jsonl trend line, errors landing in receipt["errors"]).
"""

from __future__ import annotations

import json
import re

from click.testing import CliRunner

from memo import dream_tune
from memo.cli_dream import dream_cmd
from memo.cli_dream_passes import _run_eval_recall, _run_harvest_labels


class _Hit:
    def __init__(self, id, score, title="t", tags=None, path="p", body="some body text"):
        self.id, self.score, self.title = id, score, title
        self.tags, self.path, self.body = tags or [], path, body


class _StubLifecycle:
    def enforce_forget_ttl(self, dry_run=False):
        return []


class _StubMem:
    """One relevant hit (aaaa1111 @0.9), one weaker (bbbb2222 @0.5)."""

    lifecycle = _StubLifecycle()

    def search(
        self, query, limit, mode="vec", budget_ms=None, exclude_types=None, exclude_tags=None
    ):
        return [_Hit("aaaa1111", 0.9), _Hit("bbbb2222", 0.5)]


def _write_curated(state_dir, prompts):
    p = state_dir / "eval" / "regression_labels.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")


# --- _run_harvest_labels ------------------------------------------------------


def test_harvest_creates_file_and_counts(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        "memo.eval_recall.harvest_labels",
        lambda sd, **k: [
            {"text": "how does memo sync work", "relevant": True, "expect_ids": ["aaaa1111"]}
        ],
    )
    res = _run_harvest_labels(tmp_cfg)
    assert res == {"new": 1, "total": 1}
    raw = json.loads(
        (tmp_cfg.state_dir / "eval" / "harvested_labels.json").read_text(encoding="utf-8")
    )
    assert len(raw["prompts"]) == 1
    assert raw["prompts"][0]["expect_ids"] == ["aaaa1111"]
    assert raw["prompts"][0]["harvested_ts"]  # stamped for recency capping


def test_harvest_merge_dedups_by_prompt_and_unions_ids(tmp_cfg, monkeypatch):
    monkeypatch.setattr(
        "memo.eval_recall.harvest_labels",
        lambda sd, **k: [
            {"text": "how does memo sync work", "relevant": True, "expect_ids": ["aaaa1111"]}
        ],
    )
    _run_harvest_labels(tmp_cfg)
    # Second night: same prompt with a new grounded id + one genuinely new prompt.
    monkeypatch.setattr(
        "memo.eval_recall.harvest_labels",
        lambda sd, **k: [
            {"text": "how does memo sync work", "relevant": True, "expect_ids": ["bbbb2222"]},
            {
                "text": "where does the recall daemon socket live",
                "relevant": True,
                "expect_ids": ["cccc3333"],
            },
        ],
    )
    res = _run_harvest_labels(tmp_cfg)
    assert res == {"new": 1, "total": 2}  # dup merged, new one appended
    raw = json.loads(
        (tmp_cfg.state_dir / "eval" / "harvested_labels.json").read_text(encoding="utf-8")
    )
    by_text = {p["text"]: p for p in raw["prompts"]}
    assert by_text["how does memo sync work"]["expect_ids"] == ["aaaa1111", "bbbb2222"]
    assert by_text["where does the recall daemon socket live"]["expect_ids"] == ["cccc3333"]


# --- _run_eval_recall -----------------------------------------------------------


def test_eval_recall_receipt_shape_and_history_append(tmp_cfg):
    _write_curated(
        tmp_cfg.state_dir,
        [{"text": "curated question about sync", "relevant": True, "expect_ids": ["aaaa1111"]}],
    )
    hp = tmp_cfg.state_dir / "eval" / "harvested_labels.json"
    hp.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "text": "harvested question about daemons",
                        "relevant": True,
                        "expect_ids": ["aaaa1111"],
                        "harvested_ts": "2026-07-01T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=200)
    # `*_curated` are present whenever the curated partition is non-empty: the
    # blended headline mixes ~150 harvested labels with the 46 curated ones the
    # regression discipline actually gates on.
    assert set(frag) == {
        "prec_at_k",
        "noise_at_k",
        "prec_at_k_curated",
        "noise_at_k_curated",
        "k",
        "labels_total",
        "harvested",
        "harvested_deduped_out",
        "curated",
    }
    assert frag["k"] == 3
    assert frag["labels_total"] == 2
    assert frag["harvested"] == 1
    assert frag["harvested_deduped_out"] == 0  # prompts are dissimilar — nothing absorbed
    assert frag["curated"] == 1
    assert frag["prec_at_k"] > 0
    assert frag["noise_at_k"] == 0.0

    hist = tmp_cfg.state_dir / "eval" / "history.jsonl"
    lines = hist.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "dream"
    assert entry["k"] == 3
    assert entry["labels"] == 2
    assert entry["prec_at_k"] == frag["prec_at_k"]
    assert entry["noise_at_k"] == frag["noise_at_k"]
    assert entry["ts"]

    _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=200)  # appends, not overwrites
    assert len(hist.read_text(encoding="utf-8").splitlines()) == 2


def test_eval_recall_caps_to_most_recent_harvested(tmp_cfg):
    # Curated never matches; only the NEWEST harvested label does — so a
    # nonzero precision proves the recency cap picked the right one.
    _write_curated(
        tmp_cfg.state_dir,
        [{"text": "curated question about sync", "relevant": True, "expect_ids": ["ffff9999"]}],
    )
    hp = tmp_cfg.state_dir / "eval" / "harvested_labels.json"
    hp.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "text": "old harvested question",
                        "relevant": True,
                        "expect_ids": ["eeee8888"],
                        "harvested_ts": "2026-06-01T00:00:00+00:00",
                    },
                    {
                        "text": "newest harvested question",
                        "relevant": True,
                        "expect_ids": ["aaaa1111"],
                        "harvested_ts": "2026-07-02T00:00:00+00:00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=2)
    assert frag["labels_total"] == 2
    assert frag["curated"] == 1
    assert frag["harvested"] == 1
    assert frag["prec_at_k"] > 0  # the newest (matching) label was the one kept


def test_eval_recall_cross_dedups_harvested_against_curated(tmp_cfg):
    """A harvested prompt token-Jaccard-similar (>=0.6) to a curated one is
    absorbed (counted ONCE, not double-weighted in prec@K/noise@K); the
    receipt reports the post-dedup harvested count + how many were absorbed."""
    _write_curated(
        tmp_cfg.state_dir,
        [
            {
                "text": "how does memo sync work exactly",
                "relevant": True,
                "expect_ids": ["aaaa1111"],
            }
        ],
    )
    hp = tmp_cfg.state_dir / "eval" / "harvested_labels.json"
    hp.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        # Near-dup of the curated prompt (Jaccard 5/6 >= 0.6).
                        "text": "how does memo sync work",
                        "relevant": True,
                        "expect_ids": ["bbbb2222"],
                        "harvested_ts": "2026-07-02T00:00:00+00:00",
                    },
                    {
                        # Genuinely new prompt — must survive the cross-dedup.
                        "text": "where does the recall daemon socket live",
                        "relevant": True,
                        "expect_ids": ["cccc3333"],
                        "harvested_ts": "2026-07-01T00:00:00+00:00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=200)
    assert frag["curated"] == 1
    assert frag["harvested"] == 1  # post-dedup: only the dissimilar survivor
    assert frag["harvested_deduped_out"] == 1  # the near-dup was absorbed
    assert frag["labels_total"] == 2  # counted once, not three prompts

    hist = tmp_cfg.state_dir / "eval" / "history.jsonl"
    entry = json.loads(hist.read_text(encoding="utf-8").splitlines()[0])
    assert entry["labels"] == 2


def test_eval_recall_dedup_does_not_waste_room(tmp_cfg):
    """Cross-dedup runs BEFORE the max_labels cap: an absorbed harvested
    prompt must not consume a slot a genuinely new prompt could fill."""
    _write_curated(
        tmp_cfg.state_dir,
        [
            {
                "text": "how does memo sync work exactly",
                "relevant": True,
                "expect_ids": ["aaaa1111"],
            }
        ],
    )
    hp = tmp_cfg.state_dir / "eval" / "harvested_labels.json"
    hp.write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        # Newest, but a near-dup of curated — absorbed.
                        "text": "how does memo sync work",
                        "relevant": True,
                        "expect_ids": ["bbbb2222"],
                        "harvested_ts": "2026-07-02T00:00:00+00:00",
                    },
                    {
                        # Older and distinct: with max_labels=2 there is room
                        # for exactly one harvested label — it must be this one.
                        "text": "where does the recall daemon socket live",
                        "relevant": True,
                        "expect_ids": ["aaaa1111"],
                        "harvested_ts": "2026-07-01T00:00:00+00:00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=2)
    assert frag["labels_total"] == 2
    assert frag["harvested"] == 1
    assert frag["harvested_deduped_out"] == 1
    # The surviving harvested label expects aaaa1111 (which _StubMem returns),
    # so precision proves the dedup didn't burn the room on the absorbed dup.
    assert frag["prec_at_k"] > 0


def test_eval_recall_uses_the_packaged_curated_set(tmp_cfg, tmp_path, monkeypatch):
    """Installed-runtime shape: nothing in state_dir, no repo checkout above the
    module — only the copy force-included into the wheel. The nightly receipt
    must still count curated labels.

    Before the fix this pass carried its own two-candidate lookup that omitted
    the packaged path, so `memo dream status` reported `0 curated` every night
    and prec@K (0.027 on the live corpus) was measured on harvested labels
    alone, while the tuner's gate — which does resolve the packaged copy — saw a
    different label set than the number published beside it.
    """
    packaged = tmp_path / "wheel" / "regression_labels.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text(
        json.dumps({"prompts": [{"text": "packaged curated probe", "expect_ids": ["aaaa1111"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dream_tune, "_packaged_curated_labels", lambda: packaged)
    # Kill the dev-checkout fallback so only the packaged copy can answer.
    monkeypatch.setattr(dream_tune, "__file__", str(tmp_path / "nowhere" / "dream_tune.py"))

    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=3, max_labels=200)

    assert frag["curated"] == 1


def test_eval_recall_no_labels_skips_history(tmp_cfg, monkeypatch):
    # No curated (the shared lookup finds no document) + no harvested file.
    monkeypatch.setattr(dream_tune, "curated_labels_document", lambda _sd: {})
    frag = _run_eval_recall(tmp_cfg, _StubMem(), k=5, max_labels=200)
    assert frag["labels_total"] == 0
    assert frag["prec_at_k"] == 0.0
    assert not (tmp_cfg.state_dir / "eval" / "history.jsonl").exists()


# --- dream run wiring -----------------------------------------------------------

_SKIPS = [
    "--skip-entities",
    "--skip-decay",
    "--skip-maintain",
    "--skip-orientation",
    "--skip-signal-gather",
    "--skip-prune-floor",
    "--skip-evict",
    "--skip-compress",
    "--skip-prewarm",
    "--skip-presynthesis",
]


def _dream_env(monkeypatch, tmp_path):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", "0")
    monkeypatch.setattr("memo.cli_dream._get_memory", lambda cfg: _StubMem())
    return state


def _last_receipt(state):
    return json.loads((state / "dream" / "last.json").read_text(encoding="utf-8"))


def test_dream_run_eval_flag_off_is_noop(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMO_DREAM_EVAL_ENABLED", "0")

    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = _last_receipt(state)
    assert "harvest_labels" not in receipt
    assert "eval_recall" not in receipt
    assert not (state / "eval" / "harvested_labels.json").exists()
    assert not (state / "eval" / "history.jsonl").exists()


def test_dream_run_eval_default_on_writes_receipt_and_history(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    _write_curated(
        state,
        [{"text": "curated question about sync", "relevant": True, "expect_ids": ["aaaa1111"]}],
    )
    monkeypatch.setattr(
        "memo.eval_recall.harvest_labels",
        lambda sd, **k: [
            {
                "text": "harvested question about daemons",
                "relevant": True,
                "expect_ids": ["aaaa1111"],
            }
        ],
    )

    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = _last_receipt(state)
    assert receipt["errors"] == []
    assert receipt["harvest_labels"] == {"new": 1, "total": 1}
    ev = receipt["eval_recall"]
    assert ev["k"] == 5
    assert ev["labels_total"] == 2
    assert ev["harvested"] == 1
    assert ev["curated"] == 1
    assert ev["prec_at_k"] > 0
    lines = (state / "eval" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["source"] == "dream"


def test_dream_run_eval_errors_land_in_receipt(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    _write_curated(
        state,
        [{"text": "curated question about sync", "relevant": True, "expect_ids": ["aaaa1111"]}],
    )

    def _boom(*a, **k):
        raise RuntimeError("grounding log unreadable")

    monkeypatch.setattr("memo.eval_recall.harvest_labels", _boom)

    class _BrokenSearchMem(_StubMem):
        def search(
            self, query, limit, mode="vec", budget_ms=None, exclude_types=None, exclude_tags=None
        ):
            raise RuntimeError("index locked")

    monkeypatch.setattr("memo.cli_dream._get_memory", lambda cfg: _BrokenSearchMem())

    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = _last_receipt(state)
    assert "harvest_labels" not in receipt
    assert "eval_recall" not in receipt
    errs = " | ".join(receipt["errors"])
    assert "harvest_labels: RuntimeError: grounding log unreadable" in errs
    assert "eval_recall: RuntimeError: index locked" in errs


def test_dream_status_renders_eval_lines(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    (state / "dream").mkdir(parents=True)
    (state / "dream" / "last.json").write_text(
        json.dumps(
            {
                "ts": 1751000000,
                "harvest_labels": {"new": 2, "total": 10},
                "eval_recall": {
                    "prec_at_k": 0.2,
                    "noise_at_k": 0.0,
                    "k": 5,
                    "labels_total": 42,
                    "harvested": 30,
                    "curated": 12,
                },
            }
        ),
        encoding="utf-8",
    )

    res = CliRunner().invoke(dream_cmd, ["status"])
    assert res.exit_code == 0, res.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)  # strip rich ANSI highlighting
    assert "labels harvested: +2 (total 10)" in plain
    assert "prec@5 0.2" in plain
    assert "42 labels" in plain
