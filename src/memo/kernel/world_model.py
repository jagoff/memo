"""Active World State and Belief data structures for memo kernel."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class BeliefItem:
    id: str
    topic: str
    statement: str
    confidence: float = 1.0
    status: str = "active"  # active | invalidated | disputed
    sources: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class WorldState:
    project_name: str = "default"
    beliefs: dict[str, BeliefItem] = field(default_factory=dict)
    active_task: str = ""
    code_summary: str = ""
    version: int = 1
    compiled_at: float = field(default_factory=time.time)


class WorldModel:
    """Active Latent Kernel state manager."""

    def __init__(self, state_dir: Path, project_name: str = "default") -> None:
        self.state_dir = state_dir
        self.project_name = project_name
        self.state_file = state_dir / f"world_state_{project_name}.json"
        self.state = WorldState(project_name=project_name)
        self.load()

    def load(self) -> WorldState:
        """Load world state from state_dir if present."""
        if not self.state_file.is_file():
            return self.state

        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            beliefs_dict = {}
            for bid, bdata in data.get("beliefs", {}).items():
                beliefs_dict[bid] = BeliefItem(
                    id=bdata.get("id", bid),
                    topic=bdata.get("topic", "general"),
                    statement=bdata.get("statement", ""),
                    confidence=float(bdata.get("confidence", 1.0)),
                    status=bdata.get("status", "active"),
                    sources=bdata.get("sources", []),
                    last_updated=float(bdata.get("last_updated", time.time())),
                )
            self.state = WorldState(
                project_name=data.get("project_name", self.project_name),
                beliefs=beliefs_dict,
                active_task=data.get("active_task", ""),
                code_summary=data.get("code_summary", ""),
                version=int(data.get("version", 1)),
                compiled_at=float(data.get("compiled_at", time.time())),
            )
        except Exception as exc:
            _logger.debug("Failed to load world state: %s", exc)

        return self.state

    def save(self) -> None:
        """Atomically persist world state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state.compiled_at = time.time()
        self.state.version += 1

        payload = {
            "project_name": self.state.project_name,
            "beliefs": {bid: asdict(b) for bid, b in self.state.beliefs.items()},
            "active_task": self.state.active_task,
            "code_summary": self.state.code_summary,
            "version": self.state.version,
            "compiled_at": self.state.compiled_at,
        }

        tmp_file = self.state_dir / f"{self.state_file.name}.tmp"
        try:
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(self.state_file)
        except Exception as exc:
            _logger.debug("Failed to save world state: %s", exc)

    def upsert_belief(
        self,
        belief_id: str,
        topic: str,
        statement: str,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> BeliefItem:
        """Upsert a belief into the active world model."""
        existing = self.state.beliefs.get(belief_id)
        sources = existing.sources if existing else []
        if source and source not in sources:
            sources.append(source)

        item = BeliefItem(
            id=belief_id,
            topic=topic,
            statement=statement,
            confidence=confidence,
            status="active",
            sources=sources,
            last_updated=time.time(),
        )
        self.state.beliefs[belief_id] = item
        self.save()
        return item

    def invalidate_belief(self, belief_id: str, reason: str = "") -> bool:
        """Mark a belief as invalidated."""
        if belief_id in self.state.beliefs:
            self.state.beliefs[belief_id].status = "invalidated"
            self.state.beliefs[belief_id].confidence = 0.0
            self.save()
            return True
        return False

    def get_active_beliefs(self, topic: str | None = None) -> list[BeliefItem]:
        """Return active beliefs, optionally filtered by topic."""
        active = [b for b in self.state.beliefs.values() if b.status == "active"]
        if topic:
            active = [b for b in active if b.topic == topic]
        active.sort(key=lambda b: b.confidence, reverse=True)
        return active
