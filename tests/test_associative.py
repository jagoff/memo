from memo.associative import AssociativeHit, associate


class FakeStore:
    """Entity-memory graph: mem -> entities, entity -> mems, co-recall counts."""

    def __init__(self, mem_entities: dict[str, list[str]]):
        self._me = {m: [{"name": n} for n in ns] for m, ns in mem_entities.items()}
        self._em: dict[str, list[str]] = {}
        for m, ns in mem_entities.items():
            for n in ns:
                self._em.setdefault(n, []).append(m)

    def memory_entities(self, mid: str) -> list[dict]:
        return self._me.get(mid, [])

    def entity_memories(self, name: str, type_=None) -> list[str]:
        return self._em.get(name, [])

    def co_recall_counts(self, anchor_id: str, candidate_ids: list[str]) -> dict[str, int]:
        return {}


def test_associate_finds_entity_neighbor_memory():
    # m1 mentions Memory; m2 also mentions Memory (shared entity) -> m2 is associated to m1.
    store = FakeStore({"m1": ["memory", "config"], "m2": ["memory"], "m3": ["unrelated"]})
    hits = associate(["m1"], store=store, codegraph_adj=None, exclude_ids=frozenset({"m1"}))
    ids = {h.id for h in hits}
    assert "m2" in ids
    assert "m3" not in ids
    assert all(isinstance(h, AssociativeHit) for h in hits)
    assert next(h for h in hits if h.id == "m2").via in {"memory", "config"}


def test_associate_uses_codegraph_name_join():
    # m1 mentions VecStore; codegraph says vecstore->memory; m2 mentions Memory -> linked via code.
    store = FakeStore({"m1": ["vecstore"], "m2": ["memory"]})
    cg = {"vecstore": {"memory"}, "memory": {"vecstore"}}
    hits = associate(["m1"], store=store, codegraph_adj=cg, exclude_ids=frozenset({"m1"}))
    assert "m2" in {h.id for h in hits}


def test_associate_respects_limit_and_excludes_seeds():
    store = FakeStore({"m1": ["a"], "m2": ["a"], "m3": ["a"], "m4": ["a"]})
    hits = associate(["m1"], store=store, codegraph_adj=None, limit=2,
                     exclude_ids=frozenset({"m1"}))
    assert len(hits) == 2
    assert "m1" not in {h.id for h in hits}


def test_associate_min_activation_gate_can_return_empty():
    store = FakeStore({"m1": ["a"], "m2": ["a"]})
    hits = associate(["m1"], store=store, codegraph_adj=None,
                     exclude_ids=frozenset({"m1"}), min_activation=999.0)
    assert hits == []


def test_associative_flags_registered_with_defaults():
    from memo.flags import flag_bool, flag_float, flag_int

    assert flag_bool("MEMO_RECALL_ASSOCIATIVE") is True
    assert flag_int("MEMO_ASSOCIATIVE_HOPS") == 2
    assert flag_int("MEMO_ASSOCIATIVE_LIMIT") == 2
    assert flag_float("MEMO_ASSOCIATIVE_MIN_ACTIVATION") == 0.5
    assert flag_int("MEMO_ASSOCIATIVE_BUDGET_MS") == 300


def test_associate_down_weights_hub_entities():
    # 'hub' is in many memories; 'rare' in few. A candidate reachable only via
    # the hub must score below one reachable via the rare entity.
    mem_entities = {"seed": ["hub", "rare"], "viahub": ["hub"], "viarare": ["rare"]}
    for i in range(20):
        mem_entities[f"filler{i}"] = ["hub"]
    store = FakeStore(mem_entities)
    hits = associate(["seed"], store=store, codegraph_adj=None, limit=10,
                     exclude_ids=frozenset({"seed"}))
    by_id = {h.id: h.activation for h in hits}
    assert by_id["viarare"] > by_id["viahub"]


def test_associate_hub_filtered_by_min_activation_gate():
    # A very common hub entity drops below the 0.5 gate and is dropped entirely.
    mem_entities = {"seed": ["hub"], "viahub": ["hub"]}
    for i in range(50):
        mem_entities[f"f{i}"] = ["hub"]
    store = FakeStore(mem_entities)
    hits = associate(["seed"], store=store, codegraph_adj=None,
                     exclude_ids=frozenset({"seed"}), min_activation=0.5)
    assert "viahub" not in {h.id for h in hits}


def test_associate_entity_2hop_discovery():
    # 'target' shares NO entity with the seed; it is only reachable via the
    # entity-graph 2-hop seed(a) -> bridge(a,b) -> target(b).
    store = FakeStore({"seed": ["a"], "bridge": ["a", "b"], "target": ["b"]})
    hits = associate(["seed"], store=store, codegraph_adj=None, limit=10,
                     exclude_ids=frozenset({"seed"}))
    ids = {h.id for h in hits}
    assert "target" in ids   # found via entity 2-hop, not a shared seed entity


def test_associate_overlap_ranks_multi_token_higher():
    # 'both' connects via two specific seed entities; 'one' via a single one.
    store = FakeStore({"seed": ["x", "y"], "both": ["x", "y"], "one": ["x"]})
    hits = associate(["seed"], store=store, codegraph_adj=None, limit=10,
                     exclude_ids=frozenset({"seed"}))
    by_id = {h.id: h.activation for h in hits}
    assert by_id["both"] > by_id["one"]
