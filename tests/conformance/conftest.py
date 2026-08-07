"""Corpus-scale conformance fixtures.

Seeds a deterministic ~10k-memory corpus once per session. Twelve defects found
by hand on 2026-08-06 were invisible to 6,955 unit tests because a 3-memory
fixture cannot express a page size, an unbounded neighbor list, or a total that
disagrees with the corpus. This fixture can.

Vectors come from a hash-derived stub, NOT from MLX: `MEMO_EMBEDDER_DIMS` is
pinned to `DIMS` so the store's dims guard sees a consistent profile (MLX
invariant 3). The stub validates payload size, wall-clock and total honesty --
never semantic quality, which stays with `memo eval recall` on the live corpus.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterator

import frontmatter
import pytest

from memo.config import Config
from memo.store import VecStore

DIMS = 64
TOPICS = 20

_CREATED = "2026-01-01T00:00:00+00:00"
_TOPIC_NAMES = [f"topic{n:02d}" for n in range(TOPICS)]


def seeded_id(i: int) -> str:
    """Canonical id of the i-th seeded memory. 32 hex chars, stable across runs."""
    return hashlib.sha256(f"conformance-{i}".encode()).hexdigest()[:32]


def _vector(topic: int, i: int) -> list[float]:
    """Unit vector: per-topic centroid plus small per-item jitter, so same-topic
    memories cluster and a search returns a meaningful neighborhood."""
    centroid = hashlib.sha256(f"topic-{topic}".encode()).digest()
    jitter = hashlib.sha256(f"item-{i}".encode()).digest()
    out = [
        (centroid[d % len(centroid)] - 128) / 128.0 + (jitter[d % len(jitter)] - 128) / 2048.0
        for d in range(DIMS)
    ]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


@pytest.fixture(scope="session")
def corpus_size() -> int:
    return int(os.environ.get("MEMO_CONFORMANCE_CORPUS_N", "10000"))


@pytest.fixture(scope="session")
def big_corpus(tmp_path_factory, corpus_size: int) -> Iterator[Config]:
    mp = pytest.MonkeyPatch()
    try:
        root = tmp_path_factory.mktemp("conformance")
        data, vault, state = root / "data", root / "vault", root / "state"
        for d in (data, vault, state):
            d.mkdir()

        mp.setenv("MEMO_EMBEDDER_DIMS", str(DIMS))
        mp.setenv("MEMO_NONINTERACTIVE", "1")
        mp.setenv("MEMO_DATA_DIR", str(data))
        mp.setenv("MEMO_STATE_DIR", str(state))
        mp.setenv("MEMO_AUTO_PROJECT_TAG", "0")
        mp.setenv("MEMO_STORE_BY_PROJECT", "0")

        cfg = Config(data_dir=data, vault_path=vault, state_dir=state, reranker_enabled=False)
        cfg.memory_dir.mkdir(parents=True, exist_ok=True)

        store = VecStore(cfg.db_path, dims=DIMS)
        try:
            for i in range(corpus_size):
                topic = i % TOPICS
                mid = seeded_id(i)
                title = f"Conformance memory {i} about {_TOPIC_NAMES[topic]}"
                body = (
                    f"Synthetic conformance record {i}. Subject: {_TOPIC_NAMES[topic]}. "
                    f"This body exists so body-length gates and BM25 have real text to "
                    f"work with rather than a stub token."
                )
                rel = f"{mid}.md"
                post = frontmatter.Post(
                    body,
                    id=mid,
                    title=title,
                    type="note",
                    tags=[_TOPIC_NAMES[topic], "conformance"],
                    created=_CREATED,
                    updated=_CREATED,
                    valid_at=_CREATED,
                )
                post["extra"] = {}
                post["verification_state"] = "unverified"
                (cfg.memory_dir / rel).write_text(frontmatter.dumps(post), encoding="utf-8")
                store.upsert(
                    id_=mid,
                    path=rel,
                    title=title,
                    type_="note",
                    tags=[_TOPIC_NAMES[topic], "conformance"],
                    created=_CREATED,
                    updated=_CREATED,
                    body_hash=hashlib.sha256(body.encode()).hexdigest(),
                    embedding=_vector(topic, i),
                    body_text=body,
                )
        finally:
            store.close()
    except BaseException:
        # Setup failed before the yield -- the generator never resumes, so
        # this is the only chance to revert the monkeypatched env. Without
        # it, MEMO_EMBEDDER_DIMS/MEMO_DATA_DIR/etc. leak into the rest of
        # the pytest session for every test after this one.
        mp.undo()
        raise

    yield cfg
    mp.undo()
