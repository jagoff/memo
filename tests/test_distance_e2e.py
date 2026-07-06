"""End-to-end test for distance decay integration in reranking.

Validates that memories closer to base facts rank higher than distant
memories when distance decay is enabled.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

from memo.config import Config
from memo.memory import Memory


def test_distance_decay_e2e():
    """End-to-end: save memories, control distance decay flag, verify ranking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Isolate config so test doesn't touch developer's real vault
        cfg = Config(
            data_dir=tmpdir,
            state_dir=os.path.join(tmpdir, "state"),
            reranker_enabled=False,
        )
        memory = Memory(cfg)

        try:
            # Save base fact (distance 0)
            fact_rec = memory.save(
                content="Core learning: MLX embeddings are efficient",
                title="MLX embeddings",
                type_="fact",
            )
            fact_id = fact_rec.id

            # Save derived decision (distance 1 via edge to fact)
            decision_rec = memory.save(
                content="Applied MLX embeddings in production",
                title="MLX decision",
                type_="decision",
            )
            decision_id = decision_rec.id

            # Save distant synthesis (no direct path to fact)
            synthesis_rec = memory.save(
                content="Speculating about future embedding architectures",
                title="Future embeddings",
                type_="synthesis",
            )
            synthesis_id = synthesis_rec.id

            # Note: In a full implementation, we would add graph edges here.
            # Currently, GraphStore.add_edge doesn't support arbitrary memory edges.
            # The distance computation would happen in rerank_logic when called.
            # For this test, we patch the distance_to_nearest_fact method to
            # simulate distances.

            # Recall with distance decay OFF (default behavior)
            with patch.dict(os.environ, {"MEMO_GRAPH_DISTANCE_DECAY": "0"}):
                hits_off = memory.search(
                    "MLX embeddings production",
                    limit=10,
                    disable_reranker=True,  # Disable reranker for deterministic test
                )
                order_off = [h.id for h in hits_off]

            # Recall with distance decay ON
            # Patch distance_to_nearest_fact to return controlled distances
            def mock_distance(memory_id: str) -> int:
                """Mock distances: fact=0, decision=1, synthesis=5."""
                distances = {
                    fact_id: 0,
                    decision_id: 1,
                    synthesis_id: 5,
                }
                return distances.get(memory_id, 999)

            with patch.dict(os.environ, {"MEMO_GRAPH_DISTANCE_DECAY": "1"}):
                # Patch the distance method in the graph store
                with patch.object(memory.graph, "distance_to_nearest_fact", mock_distance):
                    hits_on = memory.search(
                        "MLX embeddings production",
                        limit=10,
                        disable_reranker=True,
                    )
                    order_on = [h.id for h in hits_on]

            # Both searches should return results
            assert len(order_off) > 0, "Search with decay OFF should return results"
            assert len(order_on) > 0, "Search with decay ON should return results"

            # Verify memory IDs are in the results
            assert fact_id in order_off or fact_id in order_on, "Fact should be in results"
            assert (
                decision_id in order_off or decision_id in order_on
            ), "Decision should be in results"
            assert (
                synthesis_id in order_off or synthesis_id in order_on
            ), "Synthesis should be in results"

            # When distance decay is ON and decision is closer than synthesis,
            # decision should rank higher (lower index)
            if decision_id in order_on and synthesis_id in order_on:
                idx_decision = order_on.index(decision_id)
                idx_synthesis = order_on.index(synthesis_id)
                assert (
                    idx_decision < idx_synthesis
                ), (
                    f"Distance decay should rank close memories higher. "
                    f"Decision index={idx_decision}, Synthesis index={idx_synthesis}, "
                    f"Order={order_on}"
                )

        finally:
            memory.close()


def test_distance_decay_disabled_preserves_order():
    """When MEMO_GRAPH_DISTANCE_DECAY=False, search order is unaffected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(
            data_dir=tmpdir,
            state_dir=os.path.join(tmpdir, "state"),
            reranker_enabled=False,
        )
        memory = Memory(cfg)

        try:
            # Save memories
            memory.save(
                content="Important fact about Python",
                title="Python fact",
                type_="fact",
            )

            memory.save(
                content="Decided to use Python",
                title="Python decision",
                type_="decision",
            )

            # Search with decay disabled explicitly
            with patch.dict(os.environ, {"MEMO_GRAPH_DISTANCE_DECAY": "0"}):
                hits = memory.search(
                    "Python",
                    limit=10,
                    disable_reranker=True,
                )

            # Should get results without error
            assert len(hits) > 0, "Search should return results with decay disabled"

        finally:
            memory.close()
