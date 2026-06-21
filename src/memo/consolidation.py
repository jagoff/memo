"""Advanced consolidation — LLM-driven clustering + intelligent merge.

Expands on the basic consolidate() in memory.py by adding:
- Intelligent merge of clustered memorias into a single richer entry
- Archival of obsolete versions with forward references
- Auto-application of consolidation decisions with confirmation
- Conflict resolution when merging conflicting information

## Merge Strategy

When merging a cluster of related memorias:
1. Preserve the most recent timestamp
2. Combine tags (union, de-duplicated)
3. Merge bodies with section headers indicating source
4. Use LLM to synthesize a unified title if needed
5. Archive old memorias with a reference to the new merged one

## Archival Format

Archived memorias are moved to a subdirectory `archived/` within the memory
directory and get a frontmatter field `archived_for` pointing to the new
merged memoria ID.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from memo.llm import MLXChat
from memo.memory.record import chat_with_timeout

_log = logging.getLogger(__name__)

_MERGE_SYSTEM_PROMPT = """You merge multiple related memory notes into a single coherent entry.

You receive a cluster of 2+ memorias that have been identified as semantically
related (duplicates, evolutions, or facets). Output a JSON object:

{
  "merged_title": "concise title covering all sources",
  "merged_body": "unified body synthesizing all sources",
  "merge_strategy": "keep_latest" | "synthesis" | "concat_with_headers",
  "rationale": "1-2 sentence explanation of the merge approach"
}

Merge strategies:
- "keep_latest": the latest memoria supersedes all others (use for evolutions)
- "synthesis": create a new unified entry combining key points from all
- "concat_with_headers": concatenate with section headers per source (use for facets)

Rules:
- Preserve all factual information across sources
- Resolve conflicts by preferring the latest source unless explicitly contradicted
- Keep the merged body under 2000 chars if possible
- Output ONLY the JSON, no markdown fences, no commentary."""


@dataclass(frozen=True)
class MergeProposal:
    """A proposal for merging a cluster of memorias."""

    cluster_id: int
    memoria_ids: list[str]
    merged_title: str
    merged_body: str
    merge_strategy: str
    rationale: str
    archived_ids: list[str]  # IDs of memorias to archive after merge


@dataclass(frozen=True)
class ConsolidationResult:
    """Result of a consolidation operation."""

    merged_id: str | None  # ID of the new merged memoria (if any)
    archived_ids: list[str]  # IDs of archived memorias
    skipped_ids: list[str]  # IDs that were skipped (e.g., conflicts)
    summary: str


class AdvancedConsolidator:
    """Advanced consolidation with intelligent merge and archival.

    Args:
        memory: The Memory instance to operate on.
        chat: Optional MLXChat instance for LLM-based merge synthesis.
            If None, a new one is created on first use.
    """

    def __init__(self, memory: Any, chat: MLXChat | None = None) -> None:
        self.memory = memory
        self._chat = chat
        self._archival_dir = memory.cfg.memory_dir / "archived"

    def _ensure_chat(self) -> MLXChat:
        if self._chat is None:
            self._chat = MLXChat()
        return self._chat

    def propose_merge(
        self,
        cluster: dict[str, Any],
    ) -> MergeProposal | None:
        """Generate a merge proposal for a cluster.

        Args:
            cluster: A cluster dict from memory.consolidate(), containing
                members, summary, relationship, rationale.

        Returns:
            A MergeProposal with suggested merge strategy, or None if
            the cluster should not be merged (e.g., unrelated).
        """
        relationship = cluster.get("relationship", "unrelated")
        if relationship == "unrelated":
            return None

        members = cluster.get("members", [])
        if not members or len(members) < 2:
            return None

        # For evolutions, default to keep_latest strategy
        if relationship == "evolution":
            # Sort by updated date, take latest
            latest = max(members, key=lambda m: m.get("updated", ""))
            return MergeProposal(
                cluster_id=cluster.get("cluster_id", 0),
                memoria_ids=[m["id"] for m in members],
                merged_title=latest.get("title", ""),
                merged_body=self._read_body(latest["id"]),
                merge_strategy="keep_latest",
                rationale=f"Latest memoria ({latest['id'][:8]}) supersedes older versions",
                archived_ids=[m["id"] for m in members if m["id"] != latest["id"]],
            )

        # For duplicates and facets, use LLM to synthesize
        return self._llm_propose_merge(cluster)

    def _llm_propose_merge(self, cluster: dict[str, Any]) -> MergeProposal | None:
        """Use LLM to generate a merge proposal."""
        chat = self._ensure_chat()

        members = cluster.get("members", [])
        prompt = "Cluster of related memorias:\n\n"
        for m in members:
            prompt += f"[{m['id'][:8]}] {m['title']}\n"
            prompt += f"Updated: {m['updated']}\n"
            prompt += f"{m.get('body_preview', '')}\n\n"

        messages = [
            {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # timeout=180 must cover a COLD load of cfg.llm_model: with the 30B MoE
        # the weight read (~17GB) alone exceeds 60s, so a tighter budget would
        # kill the load before it can warm and every cluster would re-cold-load.
        # The first call warms the model; the rest reuse it in seconds.
        #
        # Greedy decode (temp=0.0) is deterministic, so a cluster whose output
        # truncates into invalid JSON fails identically on every run. One sampled
        # retry (temp=0.3) breaks that determinism and recovers it; a still-bad
        # second output degrades to a skipped cluster (never a crash).
        data: dict[str, Any] | None = None
        for attempt, temperature in enumerate((0.0, 0.3)):
            try:
                out = chat_with_timeout(
                    chat,
                    timeout=180,
                    model=self.memory.cfg.llm_model,
                    messages=messages,
                    options={"temperature": temperature, "max_tokens": 4096, "thinking": False},
                )
            except Exception as exc:
                _log.warning("consolidation: merge-proposal LLM call failed: %s", exc)
                return None
            if out is None:
                _log.warning("consolidation: merge-proposal LLM timeout")
                return None
            raw = (out.get("message") or {}).get("content") or ""
            # Decode the first JSON object from the opening brace; raw_decode
            # ignores surrounding prose / a closing fence.
            start = raw.find("{")
            if start != -1:
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(raw, start)
                    if isinstance(parsed, dict):
                        data = parsed
                        break
                except (ValueError, TypeError):
                    pass
            if attempt == 0:
                _log.info(
                    "consolidation: merge-proposal greedy output unparseable; "
                    "retrying once with sampling"
                )
        if data is None:
            _log.warning(
                "consolidation: merge-proposal JSON unparseable after retry; skipping cluster"
            )
            return None

        return MergeProposal(
            cluster_id=cluster.get("cluster_id", 0),
            memoria_ids=[m["id"] for m in members],
            merged_title=data.get("merged_title", members[0]["title"]),
            merged_body=data.get("merged_body", ""),
            merge_strategy=data.get("merge_strategy", "synthesis"),
            rationale=data.get("rationale", ""),
            archived_ids=[m["id"] for m in members],
        )

    def _read_body(self, memoria_id: str) -> str:
        """Read the body of a memoria by ID."""
        rec = self.memory.get(memoria_id)
        return rec.body if rec else ""

    def apply_merge(
        self,
        proposal: MergeProposal,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Apply a merge proposal to the corpus.

        Args:
            proposal: The MergeProposal to apply.
            dry_run: If True, don't actually modify anything, just return
                what would happen.

        Returns:
            ConsolidationResult with details of what was done.
        """
        if dry_run:
            return ConsolidationResult(
                merged_id=None,
                archived_ids=[],
                skipped_ids=proposal.memoria_ids,
                summary=f"Dry run: would merge {len(proposal.memoria_ids)} memorias",
            )

        # keep_latest: the surviving ID already exists — just archive the older ones.
        # This avoids creating a redundant copy of the latest record.
        surviving_ids = [
            mid for mid in proposal.memoria_ids if mid not in set(proposal.archived_ids)
        ]
        if proposal.merge_strategy == "keep_latest" and surviving_ids:
            surviving_id = surviving_ids[0]
            try:
                archived = []
                for mid in proposal.archived_ids:
                    if self._archive_memoria(mid, surviving_id):
                        archived.append(mid)
                return ConsolidationResult(
                    merged_id=surviving_id,
                    archived_ids=archived,
                    skipped_ids=[],
                    summary=f"Kept {surviving_id[:8]}, archived {len(archived)} superseded memorias",
                )
            except Exception as e:
                return ConsolidationResult(
                    merged_id=None,
                    archived_ids=[],
                    skipped_ids=proposal.memoria_ids,
                    summary=f"keep_latest failed: {e}",
                )

        # synthesis / concat_with_headers: create a new merged record.
        try:
            # Combine tags from all sources
            all_tags = set()
            for mid in proposal.memoria_ids:
                rec = self.memory.get(mid)
                if rec:
                    all_tags.update(rec.tags)

            # Determine type from the latest source
            latest_rec = max(
                (self.memory.get(mid) for mid in proposal.memoria_ids if self.memory.get(mid)),
                key=lambda r: r.updated if r else "",
                default=None,
            )
            type_ = latest_rec.type if latest_rec else "note"

            merged_rec = self.memory.save(
                content=proposal.merged_body,
                title=proposal.merged_title,
                type_=type_,
                tags=list(all_tags),
            )

            # Archive the old memorias
            archived = []
            for mid in proposal.archived_ids:
                if self._archive_memoria(mid, merged_rec.id):
                    archived.append(mid)

            return ConsolidationResult(
                merged_id=merged_rec.id,
                archived_ids=archived,
                skipped_ids=[],
                summary=f"Merged {len(proposal.memoria_ids)} memorias into {merged_rec.id[:8]}",
            )
        except Exception as e:
            return ConsolidationResult(
                merged_id=None,
                archived_ids=[],
                skipped_ids=proposal.memoria_ids,
                summary=f"Merge failed: {e}",
            )

    def _archive_memoria(self, memoria_id: str, replacement_id: str) -> bool:
        """Archive a memoria by moving it to the archived/ subdirectory.

        Adds a frontmatter field `archived_for` pointing to the replacement.
        """
        rec = self.memory.get(memoria_id)
        if not rec:
            return False

        # Read the original file
        source_path = self.memory._resolve_existing(rec.path)
        if not source_path.is_file():
            return False

        # Create archival directory
        self._archival_dir.mkdir(parents=True, exist_ok=True)

        # Read frontmatter
        import frontmatter

        post = frontmatter.loads(source_path.read_text(encoding="utf-8"))

        # Add archival metadata
        post["archived_for"] = replacement_id
        post["archived_at"] = datetime.now(UTC).isoformat()

        # Write to archived location
        archived_path = self._archival_dir / f"{memoria_id}.md"
        archived_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Delete from store and original location
        self.memory.delete(memoria_id)

        return True

    def _fast_lane_proposal(self, cluster: dict[str, Any]) -> MergeProposal | None:
        """Build a keep_latest MergeProposal without calling the LLM.

        Used by the high-confidence fast lane (cosine ≥ auto_threshold) where
        the similarity is high enough that LLM classification is unnecessary.
        """
        members = cluster.get("members", [])
        if not members or len(members) < 2:
            return None
        latest = max(members, key=lambda m: m.get("updated", ""))
        return MergeProposal(
            cluster_id=cluster.get("cluster_id", 0),
            memoria_ids=[m["id"] for m in members],
            merged_title=latest.get("title", ""),
            merged_body=self._read_body(latest["id"]),
            merge_strategy="keep_latest",
            rationale=f"High-confidence duplicate; keeping latest ({latest['id'][:8]})",
            archived_ids=[m["id"] for m in members if m["id"] != latest["id"]],
        )

    def consolidate_all(
        self,
        threshold: float = 0.85,
        max_clusters: int = 20,
        type_: str | None = None,
        auto_apply: bool = False,
        dry_run: bool = False,
        auto_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run full consolidation pipeline: detect clusters, propose merges, apply.

        Two-pass strategy:
        1. Fast lane (cosine ≥ auto_threshold, default 0.95): skip LLM, merge as
           keep_latest. No model cost for obviously identical memorias.
        2. Normal pass (cosine ≥ threshold, default 0.85): LLM classifies the
           relationship (duplicate / evolution / facets / unrelated). Skips any
           cluster whose members were already handled by the fast lane.

        Args:
            threshold: Cosine similarity threshold for the LLM pass.
            max_clusters: Maximum clusters per pass.
            type_: Optional filter by memoria type.
            auto_apply: If True, automatically apply all merge proposals.
            dry_run: If True, don't actually modify anything.
            auto_threshold: Override for the fast-lane threshold. If None, reads
                MEMO_CONSOLIDATE_AUTO_THRESHOLD (default 0.95).

        Returns:
            Dict with clusters, proposals, and results lists.
        """
        from memo.flags import flag_float

        if auto_threshold is None:
            auto_threshold = flag_float("MEMO_CONSOLIDATE_AUTO_THRESHOLD")

        already_merged: set[str] = set()
        all_clusters: list[dict[str, Any]] = []
        all_proposals: list[MergeProposal] = []
        all_results: list[ConsolidationResult] = []

        # Pass 1 — fast lane: high-confidence, no LLM.
        if auto_threshold is not None and auto_threshold > threshold:
            fast_clusters = self.memory.consolidate(
                threshold=auto_threshold,
                max_clusters=max_clusters,
                type_=type_,
                skip_llm=True,
            )
            all_clusters.extend(fast_clusters)
            for cluster in fast_clusters:
                proposal = self._fast_lane_proposal(cluster)
                if proposal:
                    all_proposals.append(proposal)
                    already_merged.update(proposal.memoria_ids)
                    if auto_apply:
                        all_results.append(self.apply_merge(proposal, dry_run=dry_run))

        # Pass 2 — normal pass: LLM classification for uncertain cases.
        clusters = self.memory.consolidate(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type_,
        )
        all_clusters.extend(clusters)
        for cluster in clusters:
            members = cluster.get("members", [])
            # Skip clusters already handled by the fast lane.
            if any(m["id"] in already_merged for m in members):
                continue
            proposal = self.propose_merge(cluster)
            if proposal:
                all_proposals.append(proposal)
                if auto_apply:
                    all_results.append(self.apply_merge(proposal, dry_run=dry_run))

        return {
            "clusters": all_clusters,
            "proposals": [p.__dict__ for p in all_proposals],
            "results": [r.__dict__ for r in all_results],
        }


__all__ = [
    "AdvancedConsolidator",
    "ConsolidationResult",
    "MergeProposal",
]
