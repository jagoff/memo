

def test_weighted_neighbors_is_safe_under_a_concurrent_writer(tmp_path) -> None:
    """One shared connection + a lazily-consumed cursor + another thread's
    commit is the interleaving sqlite makes no promises about.

    The read now holds the same (re-entrant) lock the writes take and
    materialises every cursor before issuing the next statement.
    """
    import threading

    from memo.graph import GraphStore

    store = GraphStore(tmp_path / "graph.db")
    try:
        for i in range(40):
            store.record_extraction(
                memory_id=f"{i:032x}",
                memory_date="2026-01-01",
                entities=[{"name": f"e{i % 7}", "type": "concept"}],
                extracted_at="2026-01-01T00:00:00+00:00",
                extractor="test",
                extractor_version="v1",
                confidence=0.9,
            )

        errors: list[BaseException] = []
        stop = threading.Event()

        def _writer() -> None:
            i = 100
            try:
                while not stop.is_set():
                    store.record_extraction(
                        memory_id=f"{i:032x}",
                        memory_date="2026-01-01",
                        entities=[{"name": f"e{i % 7}", "type": "concept"}],
                        extracted_at="2026-01-01T00:00:00+00:00",
                        extractor="test",
                        extractor_version="v1",
                        confidence=0.9,
                    )
                    i += 1
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=_writer, daemon=True)
        t.start()
        try:
            for _ in range(50):
                store.weighted_neighbors("e1")
                store.entity_names()
        finally:
            stop.set()
            t.join(timeout=10)

        assert errors == [], f"concurrent writer raised: {errors[0]!r}"
    finally:
        store.close()
