def test_topic_shift_uses_cosine_distance_when_embeddings_are_given():
    """Embeddings beat token overlap when both are available.

    Two turns can share almost no words and still be the same topic, or share
    many and not be. When the caller supplies both embeddings, the decision is
    cosine distance against `sensitivity`; token Jaccard is only the fallback.
    """
    from memo.recall_logic import detect_topic_shift

    a = [1.0, 0.0, 0.0]
    same = [1.0, 0.0, 0.0]
    orthogonal = [0.0, 1.0, 0.0]

    # Identical direction -> distance 0 -> no shift, even with zero word overlap.
    assert (
        detect_topic_shift(
            {"alfa"}, {"beta"}, sensitivity=0.35, current_embedding=a, previous_embedding=same
        )
        is False
    )
    # Orthogonal -> distance 1.0 -> a shift, even though the words overlap fully.
    assert (
        detect_topic_shift(
            {"alfa"},
            {"alfa"},
            sensitivity=0.35,
            current_embedding=a,
            previous_embedding=orthogonal,
        )
        is True
    )


def test_topic_shift_falls_back_to_tokens_on_a_degenerate_embedding():
    """A zero vector has no direction; cosine is undefined, so the token
    fallback has to take over rather than divide by zero."""
    from memo.recall_logic import detect_topic_shift

    zero = [0.0, 0.0, 0.0]
    out = detect_topic_shift(
        {"alfa", "beta"},
        {"alfa", "beta"},
        sensitivity=0.35,
        current_embedding=zero,
        previous_embedding=zero,
    )
    assert out is False  # identical token sets -> no shift
