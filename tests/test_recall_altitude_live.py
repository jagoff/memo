import inspect

from memo import recall_logic


def test_recall_logic_passes_query_to_rank_hits():
    """_recall_logic must forward the prompt as rank_hits(query=...) so altitude
    is live on the daemon path. Guard against a silent drop of the wiring."""
    src = inspect.getsource(recall_logic._recall_logic)
    assert "rank_hits(" in src
    # the prompt variable is threaded as query= into rank_hits
    assert "query=" in src


def test_rank_hits_signature_accepts_query():
    sig = inspect.signature(recall_logic.rank_hits)
    assert "query" in sig.parameters
    assert sig.parameters["query"].default is None
