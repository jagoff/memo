"""Search, evidence, artifact, and read operations for :mod:`memo.repo_index`."""

from __future__ import annotations

import shutil
from dataclasses import replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from memo.artifact_store import ContentAddressedArtifactStore
from memo.code_evidence import CodeEvidenceEnvelope, repo_corpus_evidence
from memo.embedder import assert_valid_embedding
from memo.repo_index_helpers import STATUS_INDEXING, _semantic_status
from memo.repo_index_search import (
    RepoSearchHit,
    _boost_and_resort,
    _extract_query_terms,
    _hits_from_rows,
    _rrf_fuse_repo,
    path_in_repo_scope,
)
from memo.repo_signals import collect_git_change_signals, expand_cochange_paths
from memo.repo_structural import search_codegraph_paths


class _RepoIntelligenceMixin:
    """Provider-enriched read surface mixed into ``RepoCorpus``."""

    cfg: Any
    store: Any
    embedder: Any
    last_search_diagnostics: dict[str, Any]

    def _resolve_repo_id(self, repo: str | None) -> str | None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        repo: str | None = None,
        path: str | None = None,
        mode: str = "hybrid",
        scope: str = "all",
    ) -> list[RepoSearchHit]:
        if not query or not query.strip():
            return []
        mode = str(mode or "hybrid").strip().lower()
        if mode not in {"hybrid", "lexical", "unified", "vec", "bm25", "line"}:
            raise ValueError(
                f"invalid repo search mode {mode!r}; expected hybrid, lexical, unified, "
                "vec, bm25, or line"
            )
        scope = str(scope or "all").strip().lower()
        path_in_repo_scope("", scope)
        repo_id = self._resolve_repo_id(repo) if repo else None
        if repo and repo_id is None:
            return []

        oversample = max(limit * 4, 40)
        if mode == "line":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_lines(
                        query,
                        limit=oversample,
                        repo_id=repo_id,
                        path_glob=path,
                        scope=scope,
                    )
                ),
                query=query,
                limit=limit,
            )
        if mode == "bm25":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_bm25(
                        query,
                        limit=oversample,
                        repo_id=repo_id,
                        path_glob=path,
                        scope=scope,
                    )
                ),
                query=query,
                limit=limit,
            )

        query_terms = _extract_query_terms(query)
        input_k = max(limit * 3, 40)
        bm_hits = self.store.search_repo_bm25(
            query,
            limit=input_k,
            repo_id=repo_id,
            path_glob=path,
            scope=scope,
        )
        line_hits = self.store.search_repo_lines(
            query,
            limit=input_k,
            repo_id=repo_id,
            path_glob=path,
            scope=scope,
        )
        if mode == "lexical":
            rows = _rrf_fuse_repo(
                [bm_hits, line_hits],
                channel_names=["bm25", "line"],
                limit=max(limit * 2, 30),
                query_terms=query_terms,
            )
            self.last_search_diagnostics = {
                "schema": "memo.repo_search_diagnostics.v1",
                "mode": mode,
                "scope": scope,
                "channels": {
                    "bm25": {"status": "available", "results": len(bm_hits)},
                    "line": {"status": "available", "results": len(line_hits)},
                },
            }
            return self._annotate_search_hits(
                _boost_and_resort(_hits_from_rows(rows), query=query, limit=limit)
            )

        if repo_id is not None and self.store.repo_counts(repo_id)["embedded_chunks"] == 0:
            if mode == "vec":
                return []
            rows = _rrf_fuse_repo(
                [bm_hits, line_hits],
                channel_names=["bm25", "line"],
                limit=max(limit * 2, 30),
                query_terms=query_terms,
            )
            if mode == "hybrid":
                return _boost_and_resort(_hits_from_rows(rows), query=query, limit=limit)
            return self._unified_search(
                query,
                repo_id=repo_id,
                path_glob=path,
                scope=scope,
                limit=limit,
                query_terms=query_terms,
                base_channels=[("bm25", bm_hits), ("line", line_hits)],
            )

        embedding = self.embedder.embed_query(query)
        assert_valid_embedding(embedding, self.cfg.embedder_dims, context="repo search query")
        if mode == "vec":
            return _boost_and_resort(
                _hits_from_rows(
                    self.store.search_repo_vec(
                        embedding,
                        limit=limit,
                        repo_id=repo_id,
                        path_glob=path,
                        scope=scope,
                    )
                ),
                query=query,
                limit=limit,
            )

        vec_hits = self.store.search_repo_vec(
            embedding,
            limit=input_k,
            repo_id=repo_id,
            path_glob=path,
            scope=scope,
        )
        if mode == "unified":
            return self._unified_search(
                query,
                repo_id=repo_id,
                path_glob=path,
                scope=scope,
                limit=limit,
                query_terms=query_terms,
                base_channels=[
                    ("vec", vec_hits),
                    ("bm25", bm_hits),
                    ("line", line_hits),
                ],
            )
        return _boost_and_resort(
            _hits_from_rows(
                _rrf_fuse_repo(
                    [vec_hits, bm_hits, line_hits],
                    limit=max(limit * 2, 30),
                    query_terms=query_terms,
                    channel_names=["vec", "bm25", "line"],
                )
            ),
            query=query,
            limit=limit,
        )

    def _unified_search(
        self,
        query: str,
        *,
        repo_id: str | None,
        path_glob: str | None,
        scope: str,
        limit: int,
        query_terms: list[str],
        base_channels: list[tuple[str, list[dict[str, Any]]]],
    ) -> list[RepoSearchHit]:
        sources = (
            [self.store.get_repo_source(repo_id)]
            if repo_id is not None
            else self.store.list_repo_sources(limit=100)
        )
        active_sources = [
            source
            for source in sources
            if source is not None and source.get("status") != STATUS_INDEXING
        ]
        codegraph_hits: list[dict[str, Any]] = []
        cochange_hits: list[dict[str, Any]] = []
        provider_diagnostics: dict[str, list[dict[str, Any]]] = {
            "codegraph": [],
            "cochange": [],
        }
        all_base_rows = [row for _name, rows in base_channels for row in rows]
        for source in active_sources:
            source_id = str(source["id"])
            structural = search_codegraph_paths(
                Path(str(source["clone_path"])),
                query,
                scope=scope,
                limit=max(limit * 4, 40),
            )
            structural_entries = [
                {**entry, "match_type": "codegraph"}
                for entry in structural["paths"]
                if not path_glob or fnmatch(str(entry["path"]), path_glob)
            ]
            codegraph_hits.extend(
                self.store.repo_chunks_for_paths(
                    source_id,
                    structural_entries,
                    limit=max(limit * 3, 30),
                )
            )
            provider_diagnostics["codegraph"].append(
                {
                    "repo_id": source_id,
                    "status": structural["status"],
                    "reason": structural["reason"],
                    "results": len(structural_entries),
                }
            )

            seed_paths = list(
                dict.fromkeys(
                    str(row["path"])
                    for row in all_base_rows
                    if str(row.get("repo_id") or "") == source_id
                )
            )[:20]
            signals, signal_status = self._load_change_signals(source)
            expanded = (
                expand_cochange_paths(signals, seed_paths, limit=max(limit * 4, 40))
                if signals is not None
                else []
            )
            change_entries = [
                {
                    **entry,
                    "match_type": "cochange",
                    "evidence": [
                        {
                            "kind": "git_cochange",
                            "seed_path": entry["seed_path"],
                            "count": entry["count"],
                            "confidence": entry["confidence"],
                            "cross_service": entry["cross_service"],
                            "services": entry["services"],
                        }
                    ],
                }
                for entry in expanded
                if path_in_repo_scope(str(entry["path"]), scope)
                and (not path_glob or fnmatch(str(entry["path"]), path_glob))
            ]
            cochange_hits.extend(
                self.store.repo_chunks_for_paths(
                    source_id,
                    change_entries,
                    limit=max(limit * 3, 30),
                )
            )
            provider_diagnostics["cochange"].append(
                {
                    "repo_id": source_id,
                    **signal_status,
                    "seeds": len(seed_paths),
                    "results": len(change_entries),
                }
            )

        named_channels = [
            *base_channels,
            ("codegraph", codegraph_hits),
            ("cochange", cochange_hits),
        ]
        rows = _rrf_fuse_repo(
            [channel_rows for _channel, channel_rows in named_channels],
            channel_names=[channel for channel, _rows in named_channels],
            limit=max(limit * 3, 40),
            query_terms=query_terms,
        )
        self.last_search_diagnostics = {
            "schema": "memo.repo_search_diagnostics.v1",
            "mode": "unified",
            "scope": scope,
            "channels": {
                channel: {"status": "available", "results": len(channel_rows)}
                for channel, channel_rows in base_channels
            },
            "providers": provider_diagnostics,
        }
        return self._annotate_search_hits(
            _boost_and_resort(_hits_from_rows(rows), query=query, limit=limit)
        )

    def _annotate_search_hits(self, hits: list[RepoSearchHit]) -> list[RepoSearchHit]:
        return [
            replace(
                hit,
                rank_explanation={
                    **hit.rank_explanation,
                    "search_diagnostics": self.last_search_diagnostics,
                },
            )
            for hit in hits
        ]

    def _load_change_signals(
        self,
        source: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        extra = source.get("extra")
        artifacts = extra.get("artifacts") if isinstance(extra, dict) else None
        ref = artifacts.get("change_signals") if isinstance(artifacts, dict) else None
        if not isinstance(ref, dict):
            return None, {"status": "unavailable", "reason": "artifact_missing"}
        try:
            payload = self._artifact_store().load_json(ref)
        except Exception as exc:
            return None, {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(payload, dict) or payload.get("schema") != "memo.repo_change_signals.v1":
            return None, {"status": "unavailable", "reason": "unsupported_artifact"}
        return payload, {"status": "available", "reason": ""}

    def _artifact_store(self) -> ContentAddressedArtifactStore:
        return ContentAddressedArtifactStore(Path(self.cfg.state_dir) / "artifacts")

    def _publish_repo_artifacts(
        self,
        *,
        repo_id: str,
        repo_name: str,
        clone_path: Path,
        commit_sha: str,
        generation: str,
        indexed_at: str,
        counts: dict[str, int],
        coverage_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        artifact_store = self._artifact_store()
        refs: dict[str, dict[str, Any]] = {}
        providers: dict[str, dict[str, Any]] = {}
        try:
            from memo.flags import flag_int

            max_commits = flag_int("MEMO_REPO_SIGNAL_MAX_COMMITS") or 300
            signals = collect_git_change_signals(clone_path, max_commits=max_commits)
            signal_ref = artifact_store.put_json(
                "repo-change-signals",
                signals,
                metadata={
                    "repo_id": repo_id,
                    "commit_sha": commit_sha,
                    "provider": "git-history",
                },
            )
            refs["change_signals"] = signal_ref.to_dict()
            providers["git_history"] = {
                "status": "available",
                "analyzed_commits": signals["analyzed_commits"],
                "cochange_pairs": signals["cochange_pairs"],
                "cross_service_pairs": signals["cross_service_pairs"],
            }
        except Exception as exc:
            providers["git_history"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }

        generation_ref = artifact_store.put_json(
            "repo-index-generation",
            {
                "schema": "memo.repo_index_generation.v1",
                "repo_id": repo_id,
                "repo_name": repo_name,
                "commit_sha": commit_sha,
                "index_generation": generation,
                "indexed_at": indexed_at,
                "counts": counts,
                "coverage_gaps": coverage_rows,
                "provider_artifacts": refs,
            },
            metadata={
                "repo_id": repo_id,
                "commit_sha": commit_sha,
                "index_generation": generation,
            },
        )
        refs["generation"] = generation_ref.to_dict()
        return {
            "schema": "memo.repo_artifacts.v1",
            "refs": refs,
            "providers": providers,
        }

    def evidence(
        self,
        repo: str,
        *,
        paths: list[str] | tuple[str, ...] | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
    ) -> CodeEvidenceEnvelope:
        repo_id = self._resolve_repo_id(repo)
        source = self.store.get_repo_source(repo_id) if repo_id is not None else None
        return repo_corpus_evidence(
            store=self.store,
            repo_id=repo_id,
            source=source,
            paths=paths,
            scopes=scopes,
        )

    def status(
        self,
        repo: str,
        *,
        paths: list[str] | tuple[str, ...] | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        repo_id = self._resolve_repo_id(repo)
        if repo_id is None:
            return None
        source = self.store.get_repo_source(repo_id)
        if source is None:
            return None
        counts = self.store.repo_counts(repo_id)
        semantic_status = _semantic_status(source.get("status"), counts)
        extra = source.get("extra")
        artifact_refs = extra.get("artifacts") if isinstance(extra, dict) else {}
        return {
            **source,
            **counts,
            "pending_chunks": counts["chunks"] - counts["embedded_chunks"],
            "semantic_status": semantic_status,
            "artifact_verification": {
                str(kind): self._artifact_store().verify(ref)
                for kind, ref in artifact_refs.items()
                if isinstance(ref, dict)
            }
            if isinstance(artifact_refs, dict)
            else {},
            "code_evidence": self.evidence(repo_id, paths=paths, scopes=scopes).to_dict(),
        }

    def export_artifact(
        self,
        repo: str,
        kind: str,
        destination: Path,
    ) -> dict[str, Any]:
        repo_id = self._resolve_repo_id(repo)
        source = self.store.get_repo_source(repo_id) if repo_id is not None else None
        if source is None:
            raise ValueError(f"repo not found: {repo}")
        extra = source.get("extra")
        artifacts = extra.get("artifacts") if isinstance(extra, dict) else None
        ref = artifacts.get(kind) if isinstance(artifacts, dict) else None
        if not isinstance(ref, dict):
            raise ValueError(f"artifact {kind!r} not found for repo {source['name']}")
        exported = self._artifact_store().export(ref, destination)
        return {
            "repo_id": repo_id,
            "repo_name": source["name"],
            "kind": kind,
            "digest": ref.get("digest"),
            **exported,
        }

    def get_file(
        self,
        repo: str,
        path: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> dict[str, Any] | None:
        repo_id = self._resolve_repo_id(repo)
        if repo_id is None:
            return None
        meta = self.store.get_repo_file(repo_id, path)
        if meta is None:
            return None
        lines = self.store.get_repo_file_lines(repo_id, path, start=start, end=end)
        text = "\n".join(line["text"] for line in lines)
        return {
            **meta,
            "start": lines[0]["line_no"] if lines else (start or 1),
            "end": lines[-1]["line_no"] if lines else (end or start or 1),
            "text": text,
            "lines": lines,
            "code_evidence": self.evidence(repo_id, paths=[path]).to_dict(),
        }

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_repo_sources(limit=limit)

    def delete(self, repo: str, *, remove_clone: bool = True) -> bool:
        source = self.store.get_repo_source(repo)
        if source is None:
            return False
        ok = self.store.delete_repo(source["id"])
        if ok and remove_clone:
            clone_path = source.get("clone_path")
            if clone_path:
                shutil.rmtree(clone_path, ignore_errors=True)
        return ok


__all__ = ["_RepoIntelligenceMixin"]
