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
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from memo.llm import MLXChat

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

        try:
            out = chat.chat(
                model=self.memory.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0, "max_tokens": 1024},
            )
            raw = (out.get("message") or {}).get("content") or ""
        except Exception:
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)

        try:
            data = json.loads(raw)
        except Exception:
            return None

        if not isinstance(data, dict):
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

        # Create the new merged memoria
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

    def consolidate_all(
        self,
        threshold: float = 0.85,
        max_clusters: int = 20,
        type_: str | None = None,
        auto_apply: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run full consolidation pipeline: detect clusters, propose merges, apply.

        Args:
            threshold: Cosine similarity threshold for clustering.
            max_clusters: Maximum number of clusters to process.
            type_: Optional filter by memoria type.
            auto_apply: If True, automatically apply all merge proposals.
                If False, only return proposals for review.
            dry_run: If True, don't actually modify anything.

        Returns:
            Dict with:
            - clusters: list of detected clusters
            - proposals: list of merge proposals
            - results: list of ConsolidationResult (if auto_apply)
        """
        # Step 1: Detect clusters
        clusters = self.memory.consolidate(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type_,
        )

        # Step 2: Generate merge proposals
        proposals = []
        for cluster in clusters:
            proposal = self.propose_merge(cluster)
            if proposal:
                proposals.append(proposal)

        # Step 3: Apply merges if requested
        results = []
        if auto_apply:
            for proposal in proposals:
                result = self.apply_merge(proposal, dry_run=dry_run)
                results.append(result)

        return {
            "clusters": clusters,
            "proposals": [p.__dict__ for p in proposals],
            "results": [r.__dict__ for r in results],
        }


__all__ = [
    "AdvancedConsolidator",
    "ConsolidationResult",
    "MergeProposal",
]
