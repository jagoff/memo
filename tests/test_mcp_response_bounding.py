"""Unit tests for the response-bounding helpers behind `memo_consolidate` and
emergent synthesis: `_bounded_proposal`/`_bounded_consolidate`
(`server_core_records.py`) and `_bounded_synthesis` (`server_synthesis.py`).

These are exercised end-to-end (at corpus scale) by
`tests/conformance/test_mcp_response_budget.py`, but that fixture's cluster
sizes don't reliably trip every branch (a proposal with non-list id fields, a
cluster whose member count sits exactly at the cap, a synthesis result with no
`sources` key at all). Fast, direct unit tests cover those branches without
needing the conformance corpus.
"""

from __future__ import annotations

from memo.server_core_records import _bounded_consolidate, _bounded_proposal
from memo.server_synthesis import _MAX_SYNTHESIS_SOURCES, _bounded_synthesis


def test_bounded_proposal_passes_through_non_dict():
    assert _bounded_proposal("not-a-dict", member_cap=5) == "not-a-dict"


def test_bounded_proposal_trims_both_id_lists_and_reports_true_totals():
    proposal = {
        "memory_ids": [f"m{i}" for i in range(10)],
        "archived_ids": [f"a{i}" for i in range(3)],
        "surviving_id": "m0",
    }
    out = _bounded_proposal(proposal, member_cap=4)
    assert out["memory_ids"] == [f"m{i}" for i in range(4)]
    assert out["memory_ids_shown"] == 4
    assert out["memory_ids_total"] == 10
    assert out["memory_ids_truncated"] is True
    # Under the cap: kept whole, truncated False, distinct metadata keys.
    assert out["archived_ids"] == [f"a{i}" for i in range(3)]
    assert out["archived_ids_shown"] == 3
    assert out["archived_ids_total"] == 3
    assert out["archived_ids_truncated"] is False
    assert out["surviving_id"] == "m0"


def test_bounded_proposal_ignores_non_list_id_fields():
    proposal = {"memory_ids": "not-a-list", "archived_ids": None}
    out = _bounded_proposal(proposal, member_cap=4)
    assert out == proposal


def test_bounded_consolidate_passes_through_when_clusters_missing():
    out = {"proposals": [], "results": []}
    assert _bounded_consolidate(out, cluster_limit=5, member_limit=5) is out


def test_bounded_consolidate_trims_oversized_cluster_members():
    clusters = [{"size": 8, "members": [f"id{i}" for i in range(8)]}]
    out = _bounded_consolidate({"clusters": clusters}, cluster_limit=5, member_limit=3)
    kept = out["clusters"][0]
    assert kept["members"] == ["id0", "id1", "id2"]
    assert kept["shown"] == 3
    assert kept["total"] == 8
    assert kept["truncated"] is True


def test_bounded_consolidate_leaves_cluster_under_cap_untouched():
    clusters = [{"size": 2, "members": ["id0", "id1"]}]
    out = _bounded_consolidate({"clusters": clusters}, cluster_limit=5, member_limit=3)
    assert out["clusters"][0] == {"size": 2, "members": ["id0", "id1"]}


def test_bounded_consolidate_caps_cluster_list_by_size_descending():
    clusters = [
        {"size": 1, "members": []},
        {"size": 9, "members": []},
        {"size": 5, "members": []},
    ]
    out = _bounded_consolidate({"clusters": clusters}, cluster_limit=2, member_limit=10)
    assert [c["size"] for c in out["clusters"]] == [9, 5]
    assert out["shown"] == 2
    assert out["total"] == 3
    assert out["truncated"] is True


def test_bounded_consolidate_bounds_proposals_and_their_id_lists():
    out = {
        "clusters": [],
        "proposals": [
            {"memory_ids": [f"m{i}" for i in range(6)], "archived_ids": []},
            {"memory_ids": ["m0"], "archived_ids": []},
        ],
    }
    result = _bounded_consolidate(out, cluster_limit=1, member_limit=2)
    assert result["proposals_total"] == 2
    assert result["proposals_truncated"] is True
    assert len(result["proposals"]) == 1
    # The surviving proposal's own id list was bounded too (member_limit=2).
    assert result["proposals"][0]["memory_ids"] == ["m0", "m1"]
    assert result["proposals"][0]["memory_ids_total"] == 6


def test_bounded_synthesis_trims_sources_and_reports_true_total():
    sources = [f"s{i}" for i in range(_MAX_SYNTHESIS_SOURCES + 5)]
    results = [{"id": "syn1", "sources": sources}]
    out = _bounded_synthesis(results)
    assert len(out[0]["sources"]) == _MAX_SYNTHESIS_SOURCES
    assert out[0]["total"] == len(sources)
    assert out[0]["truncated"] is True
    assert out[0]["id"] == "syn1"


def test_bounded_synthesis_passes_through_result_without_sources_list():
    results = [{"id": "syn1"}, "not-a-dict"]
    assert _bounded_synthesis(results) == results
