"""Consolidation + cross-cluster synthesis for `Memory`.

`_ConsolidateOpsMixin` holds dedup/merge consolidation and LLM cross-cluster
synthesis. They share the embedding-pull + greedy-cluster helpers, so they live
together. Extracted from maintain_ops.py; composed into `Memory` in facade.py.
"""

from __future__ import annotations

import builtins
import datetime as _dt
import json
import re as _re
from typing import Any

import frontmatter

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _CONSOLIDATE_SYSTEM_PROMPT,
    _SYNTHESIS_SYSTEM_PROMPT,
    _log,
    chat_with_timeout,
    strip_llm_output,
)


def _normalize_relative_dates(text: str, ref_date: _dt.date) -> str:
    """Replace relative temporal expressions with ISO dates anchored to ref_date.

    Never raises — returns original text on any error.
    Patterns covered (ES + EN): ayer/yesterday, hoy/today, anteayer,
    la semana pasada/last week, el mes pasado/last month,
    hace N días/N days ago.
    """
    try:
        result = text

        def _iso(d: _dt.date) -> str:
            return d.isoformat()

        # hace N días / N days ago  (before simpler patterns to avoid partial match)
        result = _re.sub(
            r"hace\s+(\d+)\s+d[ií]as?",
            lambda m: (
                f"hace {m.group(1)} días"
                f" ({_iso(ref_date - _dt.timedelta(days=int(m.group(1))))})"
            ),
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"(\d+)\s+days?\s+ago",
            lambda m: (
                f"{m.group(1)} days ago"
                f" ({_iso(ref_date - _dt.timedelta(days=int(m.group(1))))})"
            ),
            result,
            flags=_re.IGNORECASE,
        )

        # anteayer (before ayer to avoid partial match)
        result = _re.sub(
            r"\banteayer\b",
            f"anteayer ({_iso(ref_date - _dt.timedelta(days=2))})",
            result,
            flags=_re.IGNORECASE,
        )

        # ayer / yesterday
        result = _re.sub(
            r"\bayer\b",
            f"ayer ({_iso(ref_date - _dt.timedelta(days=1))})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\byesterday\b",
            f"yesterday ({_iso(ref_date - _dt.timedelta(days=1))})",
            result,
            flags=_re.IGNORECASE,
        )

        # hoy / today
        result = _re.sub(
            r"\bhoy\b",
            f"hoy ({_iso(ref_date)})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\btoday\b",
            f"today ({_iso(ref_date)})",
            result,
            flags=_re.IGNORECASE,
        )

        # la semana pasada / last week
        week_start = ref_date - _dt.timedelta(days=7)
        result = _re.sub(
            r"\bla\s+semana\s+pasada\b",
            f"la semana pasada ({_iso(week_start)})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\blast\s+week\b",
            f"last week ({_iso(week_start)})",
            result,
            flags=_re.IGNORECASE,
        )

        # el mes pasado / last month
        month_ref = ref_date.replace(day=1) - _dt.timedelta(days=1)
        month_str = month_ref.strftime("%Y-%m")
        result = _re.sub(
            r"\bel\s+mes\s+pasado\b",
            f"el mes pasado ({month_str})",
            result,
            flags=_re.IGNORECASE,
        )
        result = _re.sub(
            r"\blast\s+month\b",
            f"last month ({month_str})",
            result,
            flags=_re.IGNORECASE,
        )

        return result
    except Exception:
        return text


class _ConsolidateOpsMixin(_MemoryBase):
    def consolidate(
        self,
        *,
        threshold: float = 0.85,
        max_clusters: int = 50,
        type_: str | None = None,
        skip_llm: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Propose near-duplicate merges (LLM synthesis step).

        Optional off-resident-set path: when ``MEMO_MAINT_VIA_DAEMON=1`` and the
        maintenance daemon is reachable, the heavy synthesis LLM runs in that
        daemon's process (keeping it out of memo-mcp's resident set) and returns
        the proposals here. Any miss (flag off, daemon down) runs in-process
        exactly as before — see :meth:`_consolidate_in_process`. The daemon
        itself calls ``_consolidate_in_process`` directly, so it never re-routes
        to itself.

        When ``skip_llm=True`` the LLM classification step is skipped and all
        clusters are returned with ``relationship="duplicate"`` assumed. Used by
        the high-confidence fast lane in ``AdvancedConsolidator`` where cosine
        ≥ auto_threshold makes LLM classification unnecessary. Always runs
        in-process (daemon path is bypassed) since the O(N²) clustering is cheap
        without the LLM.
        """
        from memo.flags import flag_bool

        if not skip_llm and flag_bool("MEMO_MAINT_VIA_DAEMON"):
            from memo import maint_client

            proposals = maint_client.consolidate(
                threshold=threshold,
                max_clusters=max_clusters,
                type_=type_,
            )
            if proposals is not None:
                return proposals
            # daemon unreachable → fall through to in-process (graceful)
        return self._consolidate_in_process(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type_,
            skip_llm=skip_llm,
        )

    def _pull_embeddings(
        self,
        *,
        type_filter: str | None = None,
        exclude_types: builtins.set[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """Pull (id, embedding, title, type, tags, path, updated) for the whole
        corpus, newest-first. `type_filter` restricts to one type; `exclude_types`
        drops the given types. Direct SQL — cheaper than the per-row store API."""
        import struct

        where = ""
        params: tuple[Any, ...] = ()
        if type_filter:
            where = "WHERE meta.type = ? "
            params = (type_filter,)
        elif exclude_types:
            placeholders = ",".join("?" for _ in exclude_types)
            where = f"WHERE meta.type NOT IN ({placeholders}) "
            params = tuple(sorted(exclude_types))
        rows = self.store._conn.execute(
            "SELECT vec.id AS id, vec.embedding AS emb, "
            "       meta.title, meta.type, meta.tags, meta.path, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id " + where + "ORDER BY meta.updated DESC",
            params,
        ).fetchall()
        items: builtins.list[dict[str, Any]] = []
        for r in rows:
            blob = r["emb"]
            if not blob or len(blob) % 4 != 0:
                _log.warning("consolidate: skipping corrupt embedding for %s", r["id"][:12])
                continue
            try:
                emb = list(struct.unpack(f"<{len(blob) // 4}f", blob))
            except struct.error:
                _log.warning("consolidate: skipping corrupt embedding for %s", r["id"][:12])
                continue
            items.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "type": r["type"],
                    "tags": json.loads(r["tags"]) if r["tags"] else [],
                    "path": r["path"],
                    "updated": r["updated"],
                    "emb": emb,
                }
            )
        return items

    @staticmethod
    def _greedy_cluster(
        items: builtins.list[dict[str, Any]],
        threshold: float,
    ) -> builtins.list[builtins.list[int]]:
        """Greedy single-link clustering over L2-normalised embeddings (dot ==
        cosine). Uses numpy for the O(N²) pass when available (1024-dim × 2000
        in pure Python ≈ 400s; numpy < 1s), else a pure-Python fallback."""
        try:
            import numpy as _np

            _mat = _np.array([it["emb"] for it in items], dtype=_np.float32)
            _norms = _np.linalg.norm(_mat, axis=1, keepdims=True)
            _norms[_norms == 0] = 1.0
            _mat = _mat / _norms
            _reps: builtins.list[int] = []
            _cluster_map: builtins.list[int] = [-1] * len(items)
            for i in range(len(items)):
                if _reps:
                    sims = _mat[_reps] @ _mat[i]
                    best = int(_np.argmax(sims))
                    if float(sims[best]) >= threshold:
                        _cluster_map[i] = _reps[best]
                        continue
                _reps.append(i)
                _cluster_map[i] = i
            _cluster_dict: dict[int, builtins.list[int]] = {}
            for i, rep in enumerate(_cluster_map):
                _cluster_dict.setdefault(rep, []).append(i)
            return list(_cluster_dict.values())
        except ImportError:
            clusters: builtins.list[builtins.list[int]] = []
            for i in range(len(items)):
                joined = False
                for cluster in clusters:
                    rep_item = items[cluster[0]]
                    if (
                        sum(x * y for x, y in zip(items[i]["emb"], rep_item["emb"], strict=True))
                        >= threshold
                    ):
                        cluster.append(i)
                        joined = True
                        break
                if not joined:
                    clusters.append([i])
            return clusters

    def _consolidate_in_process(
        self,
        *,
        threshold: float = 0.85,
        max_clusters: int = 50,
        type_: str | None = None,
        skip_llm: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Find clusters of near-duplicate memorias and propose actions.

        Algorithm:
        1. Pull all stored embeddings (we have them already; no re-embed).
        2. Greedy single-link clustering by cosine ≥ `threshold`.
           Each memoria joins the first existing cluster it's
           ≥-similar to, or starts a new one.
        3. Drop singletons. The remaining clusters are candidates.
        4. For each cluster, MLXChat 7B reads the bodies and emits a
           JSON `{summary, relationship, rationale}` per
           `_CONSOLIDATE_SYSTEM_PROMPT`.
        5. Return ranked clusters (largest first), capped at
           `max_clusters` to keep the LLM cost finite on big corpora.

        DOES NOT modify anything. The user reviews the output and
        decides via `memo update` / `memo delete`.

        Threshold tuning: 0.85 catches obvious dupes, 0.92+ only catches
        near-identical text. The default 0.85 is conservative for the
        Qwen3-Embedding-0.6B vector space.
        """
        # 1) Pull all embeddings (already stored; no re-embed) and
        # 2) greedy single-link cluster by cosine ≥ threshold.
        items = self._pull_embeddings(type_filter=type_)
        if not items:
            return []
        clusters = self._greedy_cluster(items, threshold)

        # 3) Drop singletons; rank by size (then by most-recent updated).
        candidate_clusters = [c for c in clusters if len(c) >= 2]
        candidate_clusters.sort(
            key=lambda c: (-len(c), items[c[0]]["updated"]),
        )
        candidate_clusters = candidate_clusters[:max_clusters]

        if not candidate_clusters:
            return []

        # 4) For each cluster, ask MLXChat to summarise + classify.
        #    Skipped when skip_llm=True (fast lane at very high cosine threshold).
        out: list[dict[str, Any]] = []
        for ci, cluster in enumerate(candidate_clusters):
            members = []
            for idx in cluster:
                it = items[idx]
                body = self._read_body(it["path"])
                members.append(
                    {
                        "id": it["id"],
                        "id_short": it["id"][:8],
                        "title": it["title"],
                        "type": it["type"],
                        "tags": it["tags"],
                        "updated": it["updated"],
                        "body_preview": (body[:600] + ("…" if len(body) > 600 else "")),
                    }
                )

            if skip_llm:
                out.append(
                    {
                        "cluster_id": ci,
                        "size": len(members),
                        "members": members,
                        "summary": "",
                        "relationship": "duplicate",
                        "rationale": f"High-confidence cluster (cosine ≥ {threshold:.2f}); LLM skipped.",
                    }
                )
                continue

            # Build LLM prompt with all members, capped to avoid blowing the
            # context window on large clusters (50+ items × 600 chars each).
            _MAX_CONSOLIDATION_PROMPT_CHARS = 24_000
            member_lines = [
                f"[{m['id_short']}] title: {m['title']}  |  updated: {m['updated']}\n"
                f"{m['body_preview']}"
                for m in members
            ]
            _chars = 0
            _included = []
            for _line in member_lines:
                if _chars + len(_line) > _MAX_CONSOLIDATION_PROMPT_CHARS:
                    break
                _included.append(_line)
                _chars += len(_line) + 5  # 5 for "\n---\n" separator
            prompt = "Cluster:\n\n" + "\n---\n".join(_included)
            try:
                chat = self._ensure_chat()
                chat_out = chat_with_timeout(
                    chat,
                    timeout=180,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _CONSOLIDATE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384, "thinking": False},
                )
                if chat_out is None:
                    _log.warning("consolidate: LLM timeout for cluster %d", ci)
                    continue
                text = ((chat_out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("consolidate: LLM call failed for cluster %d: %s", ci, exc)
                text = ""
            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                data = {}
            out.append(
                {
                    "cluster_id": ci,
                    "size": len(members),
                    "members": members,
                    "summary": (data.get("summary") or "").strip(),
                    "relationship": data.get("relationship")
                    if data.get("relationship") in ("duplicate", "evolution", "facets", "unrelated")
                    else "unrelated",
                    "rationale": (data.get("rationale") or "").strip(),
                }
            )
        return out

    def synthesize_cross_cluster(
        self,
        *,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
        min_confidence: str | None = None,
        dry_run: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Generate emergent insights from semantically related memory clusters.

        Unlike consolidation (which asks "are these the same thing?"), synthesis
        asks "what do these memories collectively IMPLY that none states alone?"
        Results are saved as ``type=synthesis`` memories with provenance links.

        Algorithm:
        1. Cluster all durable (non-reference, non-synthesis) memories at a
           looser cosine threshold than consolidation (default 0.78 vs 0.85).
        2. Drop clusters smaller than ``min_cluster_size`` (default 3).
        3. For each cluster, check if an up-to-date synthesis already exists
           (provenance hash match) — skip if so.
        4. Call MLXChat with ``_SYNTHESIS_SYSTEM_PROMPT``. If confidence meets
           the floor and title is non-null, save a new ``type=synthesis`` record.
        5. Return the list of results (proposed or saved).

        With ``dry_run=True``, returns proposals without saving anything.
        """
        import hashlib

        from memo.flags import flag_float, flag_int, flag_str

        threshold = (
            threshold if threshold is not None else (flag_float("MEMO_SYNTHESIS_THRESHOLD") or 0.78)
        )
        min_cluster_size = (
            min_cluster_size
            if min_cluster_size is not None
            else (flag_int("MEMO_SYNTHESIS_MIN_CLUSTER") or 3)
        )
        max_clusters = (
            max_clusters
            if max_clusters is not None
            else (flag_int("MEMO_SYNTHESIS_MAX_CLUSTERS") or 20)
        )
        min_confidence = min_confidence or flag_str("MEMO_SYNTHESIS_MIN_CONFIDENCE") or "medium"
        _conf_rank = {"low": 0, "medium": 1, "high": 2}
        min_conf_rank = _conf_rank.get(min_confidence, 1)

        store_conn = self.store._conn

        # 1) Pull all non-synthesis, non-reference embeddings and
        # 2) greedy single-link cluster (shared with consolidation; numpy fast
        #    path applies here too now).
        items = self._pull_embeddings(exclude_types={"reference", "synthesis"})
        if not items:
            return []
        clusters = self._greedy_cluster(items, threshold)

        candidate_clusters = [c for c in clusters if len(c) >= min_cluster_size]
        candidate_clusters.sort(key=lambda c: (-len(c), items[c[0]]["updated"]))
        candidate_clusters = candidate_clusters[:max_clusters]

        if not candidate_clusters:
            return []

        # 3) Load existing synthesis provenance hashes to skip duplicates.
        existing_hashes: set[str] = set()
        existing_rows = store_conn.execute(
            "SELECT meta.path FROM meta WHERE meta.type = 'synthesis'",
        ).fetchall()
        for er in existing_rows:
            p = er["path"]
            if p:
                ep = self._resolve_existing(p)
                if ep.is_file():
                    try:
                        ep_post = frontmatter.loads(ep.read_text(encoding="utf-8"))
                        _ep_extra: dict = ep_post.get("extra") or {}  # type: ignore[assignment]
                        h = str(_ep_extra.get("synthesis_sources_hash") or "").strip()
                        if h:
                            existing_hashes.add(h)
                    except (OSError, ValueError) as exc:
                        # A bad existing synthesis file silently missed its hash →
                        # duplicate synthesis on the next run. Log the breadcrumb.
                        _log.debug("synthesis: could not read hash from %s: %s", p, exc)

        # 4) LLM synthesis pass.
        chat = self._ensure_chat()

        out: list[dict[str, Any]] = []
        saved = 0

        for cluster in candidate_clusters:
            source_ids = [items[idx]["id"] for idx in cluster]
            sources_hash = hashlib.sha256(",".join(sorted(source_ids)).encode()).hexdigest()[:16]

            if sources_hash in existing_hashes:
                continue  # up-to-date synthesis already exists

            members = []
            for idx in cluster:
                it = items[idx]
                body = self._read_body(it["path"])
                members.append(
                    {
                        "id": it["id"],
                        "id_short": it["id"][:8],
                        "title": it["title"],
                        "type": it["type"],
                        "body_preview": (body[:500] + ("…" if len(body) > 500 else "")),
                    }
                )

            _MAX_CHARS = 20_000
            lines = [
                f"[{m['id_short']}] ({m['type']}) {m['title']}\n{m['body_preview']}"
                for m in members
            ]
            _chars = 0
            _included: list[str] = []
            for line in lines:
                if _chars + len(line) > _MAX_CHARS:
                    break
                _included.append(line)
                _chars += len(line) + 5

            prompt = "Cluster:\n\n" + "\n---\n".join(_included)

            try:
                chat_out = chat_with_timeout(
                    chat,
                    timeout=180,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0, "max_tokens": 512, "thinking": False},
                )
                if chat_out is None:
                    _log.warning("synthesize: LLM timeout for cluster")
                    continue
                text = ((chat_out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("synthesize: LLM call failed for cluster: %s", exc)
                continue

            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                data = {}

            title = (data.get("title") or "").strip()
            body = (data.get("body") or "").strip()
            # normalize relative temporal references to ISO dates
            body = _normalize_relative_dates(body, _dt.date.today())
            confidence = data.get("confidence") if data.get("confidence") in _conf_rank else "low"
            rationale = (data.get("rationale") or "").strip()

            result: dict[str, Any] = {
                "sources": source_ids,
                "sources_hash": sources_hash,
                "title": title,
                "body": body,
                "confidence": confidence,
                "rationale": rationale,
                "saved": False,
            }

            if (
                title
                and body
                and _conf_rank.get(str(confidence), 0) >= min_conf_rank
                and not dry_run
            ):
                try:
                    rec = self.save(
                        content=body,
                        title=title,
                        type_="synthesis",
                        tags=["synthesis"],
                        extra={
                            "synthesis_sources": source_ids,
                            "synthesis_sources_hash": sources_hash,
                            "synthesis_rationale": rationale,
                            "synthesis_confidence": confidence,
                        },
                    )
                    result["saved"] = True
                    result["id"] = rec.id
                    existing_hashes.add(sources_hash)
                    saved += 1
                except Exception as exc:
                    _log.warning("synthesize: save failed: %s", exc)

            out.append(result)

        _log.info("synthesize: processed %d clusters → %d saved", len(out), saved)
        return out
