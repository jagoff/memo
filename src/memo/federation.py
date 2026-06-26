"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Multi-vault federation — search across multiple memo vaults.

Enables searching and aggregating results from multiple memo vaults:
- Configure vault connections (path + name)
- Search across all configured vaults
- Aggregate results with deduplication
- Per-vault filtering and weighting
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Configuration for a single vault in federation."""

    name: str
    path: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class FederatedResult:
    """A search result from federation."""

    memoria_id: str
    vault_name: str
    title: str
    body: str
    score: float
    type: str
    tags: list[str]


class FederationConfig:
    """Manages federation vault configuration.

    Args:
        config_path: Path to the federation config JSON file.
    """

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._vaults: dict[str, VaultConfig] = {}
        self._load()

    def _load(self) -> None:
        """Load vaults from config file."""
        if self.config_path.is_file():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                for name, v in data.get("vaults", {}).items():
                    vault_data = dict(v)
                    vault_data.pop("name", None)
                    self._vaults[name] = VaultConfig(name=name, **vault_data)
            except (OSError, ValueError, TypeError) as exc:
                _log.warning("federation: config unreadable, starting empty: %s", exc)
                self._vaults = {}

    def _save(self) -> None:
        """Save vaults to config file."""
        try:
            data = {
                "vaults": {
                    name: {
                        "path": v.path,
                        "weight": v.weight,
                        "enabled": v.enabled,
                    }
                    for name, v in self._vaults.items()
                }
            }
            self.config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            _log.error("federation: failed to persist vault config: %s", exc)

    def add_vault(self, name: str, path: str, weight: float = 1.0) -> None:
        """Add a vault to the federation.

        Args:
            name: Vault name (unique).
            path: Absolute path to vault data_dir.
            weight: Weight for result ranking (default: 1.0).
        """
        self._vaults[name] = VaultConfig(name=name, path=path, weight=weight)
        self._save()

    def remove_vault(self, name: str) -> bool:
        """Remove a vault from the federation.

        Args:
            name: Vault name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._vaults:
            del self._vaults[name]
            self._save()
            return True
        return False

    def get_vault(self, name: str) -> VaultConfig | None:
        """Get a vault configuration by name."""
        return self._vaults.get(name)

    def list_vaults(self) -> list[VaultConfig]:
        """List all configured vaults."""
        return list(self._vaults.values())

    def get_enabled_vaults(self) -> list[VaultConfig]:
        """Get only enabled vaults."""
        return [v for v in self._vaults.values() if v.enabled]


class FederationSearcher:
    """Searches across multiple vaults in federation.

    Args:
        config: The FederationConfig with vault configurations.
    """

    def __init__(self, config: FederationConfig) -> None:
        self.config = config

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[FederatedResult]:
        """Search across all enabled vaults.

        Args:
            query: Search query.
            limit: Max results per vault before aggregation.
            mode: Search mode (vec, bm25, hybrid).

        Returns:
            List of FederatedResult objects, aggregated and deduplicated.
        """
        vaults = self.config.get_enabled_vaults()
        all_results: list[FederatedResult] = []

        for vault in vaults:
            try:
                vault_results = self._search_vault(vault, query, limit, mode)
                all_results.extend(vault_results)
            except Exception:  # noqa: S112
                # Skip vaults that fail to load/search
                continue

        # Deduplicate and re-rank
        deduped = self._deduplicate_results(all_results)
        ranked = self._rank_results(deduped)

        return ranked[:limit]

    def _search_vault(
        self,
        vault: VaultConfig,
        query: str,
        limit: int,
        mode: str,
    ) -> list[FederatedResult]:
        """Search a single vault.

        Args:
            vault: Vault configuration.
            query: Search query.
            limit: Result limit.
            mode: Search mode.

        Returns:
            List of FederatedResult objects from this vault.
        """
        # Import here to avoid circular dependency
        from memo.config import Config
        from memo.memory import Memory

        # Create a Memory instance for this vault
        vault_cfg = Config(data_dir=Path(vault.path))
        vault_mem = Memory(vault_cfg)

        # Search the vault
        hits = vault_mem.search(query, limit=limit, mode=mode)

        # Convert to FederatedResult
        results = []
        for hit in hits:
            # Apply vault weight to score
            weighted_score = (hit.score or 0.0) * vault.weight

            results.append(
                FederatedResult(
                    memoria_id=hit.id,
                    vault_name=vault.name,
                    title=hit.title,
                    body=hit.body or "",
                    score=weighted_score,
                    type=hit.type,
                    tags=hit.tags,
                )
            )

        return results

    def _deduplicate_results(self, results: list[FederatedResult]) -> list[FederatedResult]:
        """Deduplicate results by memory ID.

        If the same memory ID appears from multiple vaults, keep the one
        with the highest score.

        Args:
            results: All results from all vaults.

        Returns:
            Deduplicated list of results.
        """
        seen: dict[str, FederatedResult] = {}

        for result in results:
            key = result.memoria_id
            if key not in seen or result.score > seen[key].score:
                seen[key] = result

        return list(seen.values())

    def _rank_results(self, results: list[FederatedResult]) -> list[FederatedResult]:
        """Rank results by score descending.

        Args:
            results: Deduplicated results.

        Returns:
            Sorted list of results.
        """
        return sorted(results, key=lambda r: r.score, reverse=True)


__all__ = [
    "FederatedResult",
    "FederationConfig",
    "FederationSearcher",
    "VaultConfig",
]
