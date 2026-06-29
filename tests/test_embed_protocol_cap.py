def test_max_line_bytes_fits_largest_batch():
    from memo.embed_protocol import MAX_LINE_BYTES

    # 4096 dims * 32 chunk * ~20 bytes/float JSON must fit
    assert MAX_LINE_BYTES >= 4096 * 32 * 20
