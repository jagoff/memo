"""Tests for memo.capture_core — core capture pipeline functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from memo.capture_core import (
    _extract_text,
    _hash_assistant,
    _jaccard,
    _passes_prefilter,
    _passes_quality,
    _strip_private,
    collect_tool_files,
    dedupe_batch,
    extract_and_save_text,
    extract_insights,
    find_near_duplicate,
    is_meta_commentary,
    is_near_duplicate,
    score_type_confidence,
    strip_meta_commentary,
)


class TestTranscriptParsing:
    """Tests for transcript parsing utilities."""

    def test_extract_text_string(self) -> None:
        """Simple string content extraction."""
        text = _extract_text("hello world")
        assert text == "hello world"

    def test_extract_text_none(self) -> None:
        """None content returns empty string."""
        text = _extract_text(None)
        assert text == ""

    def test_extract_text_list_with_text_blocks(self) -> None:
        """Extract text from list of content blocks."""
        content = [
            {"type": "text", "text": "first block"},
            {"type": "text", "text": "second block"},
        ]
        text = _extract_text(content)
        assert text == "first block\n\nsecond block"

    def test_extract_text_skips_non_text_blocks(self) -> None:
        """Non-text blocks (e.g., images) are skipped."""
        content = [
            {"type": "text", "text": "keep this"},
            {"type": "image", "url": "..."},
            {"type": "text", "text": "keep this too"},
        ]
        text = _extract_text(content)
        assert "keep this" in text
        assert "image" not in text

    def test_extract_text_survives_dict_valued_text_block(self) -> None:
        """A malformed text block with a non-str 'text' must not crash the pass
        ('dict' object has no attribute 'strip') — it is skipped, siblings kept."""
        content = [
            {"type": "text", "text": "good block"},
            {"type": "text", "text": {"nested": "malformed"}},
            {"type": "text", "text": "another good block"},
        ]
        text = _extract_text(content)
        assert text == "good block\n\nanother good block"


class TestMetaCommentaryFilter:
    """Tests for meta-commentary and LLM filler detection."""

    def test_is_meta_commentary_voy_a(self) -> None:
        """Spanish 'voy a' pattern is meta-commentary."""
        assert is_meta_commentary("voy a implementar la solución")

    def test_is_meta_commentary_let_me(self) -> None:
        """English 'let me' pattern is meta-commentary."""
        assert is_meta_commentary("let me check the file")

    def test_is_meta_commentary_false_for_substance(self) -> None:
        """Substantial text is not meta-commentary."""
        assert not is_meta_commentary("I decided to use Redis")

    def test_strip_meta_commentary_narration_only(self) -> None:
        """Pure narration text is dropped entirely."""
        text = "let me implement the fix"
        result = strip_meta_commentary(text)
        assert result == ""

    def test_strip_meta_commentary_partial(self) -> None:
        """Filler at start is stripped; substance is kept."""
        text = "Okay, the fix is to use flock"
        result = strip_meta_commentary(text)
        assert result == "the fix is to use flock"

    def test_strip_meta_commentary_multiple_sentences(self) -> None:
        """Multiple sentences: narration dropped, substance kept."""
        text = "let me check the file. The issue is in line 42."
        result = strip_meta_commentary(text)
        assert "let me check" not in result
        assert "issue is in line 42" in result


class TestTypeConfidenceScoring:
    """Tests for type classification confidence."""

    def test_score_type_confidence_with_markers(self) -> None:
        """Type with matching markers scores high."""
        # "decided", "preference", etc. are markers for "decision" type
        text = "I decided to use Qwen3-0.6B for efficiency."
        score = score_type_confidence("decision", text)
        assert score >= 0.7

    def test_score_type_confidence_no_markers(self) -> None:
        """Type without own markers scores lower."""
        text = "Just a note about the session"
        score = score_type_confidence("note", text)
        # Note type has no specific markers, so neutral/low
        assert 0.3 <= score <= 0.7

    def test_score_type_confidence_no_own_markers_with_others(self) -> None:
        """Type with no own markers but other type's markers present."""
        # "bug" type has markers like "bug", "error", "broke"
        # If we claim "note" (which has no markers) but text contains bug-related markers
        text = "there was a bug in the regex pattern that I fixed"
        score = score_type_confidence("note", text)
        # Note type has no markers of its own
        # Whether others are present affects the score (0.6 if none, 0.4 if some)
        # This is checking behavior, not a specific value
        assert 0.3 <= score <= 0.7


class TestQualityGating:
    """Tests for quality gate logic."""

    def test_passes_quality_too_short(self) -> None:
        """Text shorter than min_words fails quality."""
        text = "too short"
        assert not _passes_quality(text, min_words=10)

    def test_passes_quality_sufficient_length(self) -> None:
        """Text with sufficient length passes quality."""
        text = "I decided to use Qwen3-0.6B for better accuracy in embeddings"
        assert _passes_quality(text, min_words=5)

    def test_passes_quality_generic_prefix(self) -> None:
        """Generic session-narrative openers fail quality."""
        text = "the user asked me to implement a fix"
        assert not _passes_quality(text, min_words=5)

    def test_passes_quality_temporal_noise(self) -> None:
        """Pure temporal markers fail quality."""
        text = "on 2026-07-01 the system was deployed"
        assert not _passes_quality(text, min_words=5)


class TestPreFilter:
    """Tests for pre-filter (cheap keyword check)."""

    def test_passes_prefilter_with_trigger_keyword(self) -> None:
        """Text with trigger keyword passes."""
        text = "A" * 200 + " decided to use PostgreSQL"
        assert _passes_prefilter(text)

    def test_passes_prefilter_too_short(self) -> None:
        """Text shorter than min_chars fails."""
        text = "decided"
        assert not _passes_prefilter(text)

    def test_passes_prefilter_no_keyword(self) -> None:
        """Text without trigger keyword fails."""
        text = "A" * 200 + " the session continued without incident"
        assert not _passes_prefilter(text)

    def test_passes_prefilter_case_insensitive(self) -> None:
        """Keyword check is case-insensitive."""
        text = "A" * 200 + " DECIDED to use this pattern"
        assert _passes_prefilter(text)


class TestPrivateMarkers:
    """Tests for <private> block stripping."""

    def test_strip_private_no_markers(self) -> None:
        """Text without <private> markers is unchanged."""
        text = "keep this entire message"
        assert _strip_private(text) == text

    @patch("memo.flags.flag_bool")
    def test_strip_private_markers_disabled(self, mock_flag: MagicMock) -> None:
        """When disabled, <private> blocks are kept."""
        mock_flag.return_value = False
        text = "<private>secret</private> public"
        # When disabled, strip_private_spans is not called
        result = _strip_private(text)
        assert result == text


class TestToolActivityProjection:
    """Tests for tool activity collection."""

    def test_collect_tool_files_write_tools(self, tmp_path: Path) -> None:
        """Write tool calls are tracked."""
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text(
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "/src/foo.py"},
                        }
                    ]
                },
            })
            + "\n"
        )
        files = collect_tool_files(transcript_file)
        assert "/src/foo.py" in files["files_modified"]

    def test_collect_tool_files_read_tools(self, tmp_path: Path) -> None:
        """Read tool calls are tracked separately."""
        transcript_file = tmp_path / "test.jsonl"
        transcript_file.write_text(
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/src/foo.py"},
                        }
                    ]
                },
            })
            + "\n"
        )
        files = collect_tool_files(transcript_file)
        assert "/src/foo.py" in files["files_read"]

    def test_collect_tool_files_empty_transcript(self, tmp_path: Path) -> None:
        """Empty transcript returns empty arrays."""
        transcript_file = tmp_path / "empty.jsonl"
        transcript_file.touch()
        files = collect_tool_files(transcript_file)
        assert files["files_read"] == []
        assert files["files_modified"] == []


class TestDeduplication:
    """Tests for near-duplicate detection and batch dedup."""

    def test_jaccard_identical_strings(self) -> None:
        """Jaccard of identical strings is 1.0."""
        score = _jaccard("hello world", "hello world")
        assert score == 1.0

    def test_jaccard_disjoint_strings(self) -> None:
        """Jaccard of disjoint strings is 0.0."""
        score = _jaccard("hello", "goodbye")
        assert score == 0.0

    def test_jaccard_partial_overlap(self) -> None:
        """Jaccard of partial overlap is between 0 and 1."""
        score = _jaccard("hello world foo", "hello world bar")
        # Overlap: {hello, world}, union: {hello, world, foo, bar}
        # Jaccard = 2/4 = 0.5
        assert abs(score - 0.5) < 0.01

    @patch("memo.capture_core._log")
    def test_find_near_duplicate_no_embedder(self, mock_log: MagicMock) -> None:
        """When embedder fails, find_near_duplicate returns None."""
        mem = MagicMock()
        mem.embedder.embed_query.side_effect = Exception("MLX down")
        candidate = {"title": "test", "body": "test body"}
        result = find_near_duplicate(mem, candidate)
        assert result is None

    def test_is_near_duplicate_wrapper(self) -> None:
        """is_near_duplicate wraps find_near_duplicate."""
        mem = MagicMock()
        mem.embedder.embed_query.side_effect = Exception("MLX down")
        candidate = {"title": "test", "body": "test body"}
        result = is_near_duplicate(mem, candidate)
        assert result is False

    @patch("memo.capture_core._log")
    def test_dedupe_batch_single_candidate(self, mock_log: MagicMock) -> None:
        """Single candidate batch returns candidate unchanged."""
        candidates = [{"title": "test", "body": "content"}]
        mem = MagicMock()
        kept, dropped = dedupe_batch(candidates, mem, 0.85)
        assert kept == candidates
        assert dropped == 0

    @patch("memo.capture_core._log")
    def test_dedupe_batch_empty(self, mock_log: MagicMock) -> None:
        """Empty batch returns empty."""
        kept, dropped = dedupe_batch([], MagicMock(), 0.85)
        assert kept == []
        assert dropped == 0


class TestExtractInsights:
    """Tests for LLM extraction pipeline."""

    def test_extract_insights_empty_response(self) -> None:
        """Empty response returns empty list."""
        helper = MagicMock()
        helper.chat.return_value = {"message": {"content": ""}}
        result = extract_insights(helper, "test-model", "user", "assistant")
        assert result == []

    def test_extract_insights_invalid_json(self) -> None:
        """Invalid JSON response returns empty list."""
        helper = MagicMock()
        helper.chat.return_value = {"message": {"content": "not json"}}
        result = extract_insights(helper, "test-model", "user", "assistant")
        assert result == []

    def test_extract_insights_skips_non_string_fields(self) -> None:
        """LLM items with nested dict/list title/body/type are skipped, not crashed on."""
        import json as _json

        helper = MagicMock()
        payload = [
            {"title": "ok", "body": {"nested": "dict"}, "type": "note"},
            {"title": ["list"], "body": "b", "type": "note"},
            {"title": "kept", "body": "real body", "type": "note"},
        ]
        helper.chat.return_value = {"message": {"content": _json.dumps(payload)}}
        result = extract_insights(helper, "test-model", "user", "assistant")
        assert [i["title"] for i in result] == ["kept"]

    def test_extract_insights_valid_extraction(self) -> None:
        """Valid extraction returns parsed insights."""
        helper = MagicMock()
        insights_json = json.dumps([
            {
                "title": "Use PostgreSQL",
                "type": "decision",
                "body": "Decided to use PostgreSQL for reliability.",
                "tags": ["database", "decision"],
            }
        ])
        helper.chat.return_value = {"message": {"content": insights_json}}
        result = extract_insights(helper, "test-model", "user text", "assistant text")
        assert len(result) == 1
        assert result[0]["title"] == "Use PostgreSQL"

    def test_extract_insights_fenced_response(self) -> None:
        """Handles markdown-fenced JSON responses."""
        helper = MagicMock()
        insights_json = json.dumps([
            {"title": "test", "type": "note", "body": "content", "tags": []}
        ])
        fenced = f"```json\n{insights_json}\n```"
        helper.chat.return_value = {"message": {"content": fenced}}
        result = extract_insights(helper, "test-model", "user", "assistant")
        assert len(result) == 1

    def test_extract_insights_helper_failure(self) -> None:
        """Helper LLM failure is absorbed."""
        helper = MagicMock()
        helper.chat.side_effect = Exception("LLM error")
        result = extract_insights(helper, "test-model", "user", "assistant")
        assert result == []


class TestHashAndExtractSave:
    """Tests for hashing and the extraction+save pipeline."""

    def test_hash_assistant_consistency(self) -> None:
        """Same text produces same hash."""
        text = "assistant response"
        h1 = _hash_assistant(text)
        h2 = _hash_assistant(text)
        assert h1 == h2

    def test_hash_assistant_different(self) -> None:
        """Different texts produce different hashes."""
        h1 = _hash_assistant("text1")
        h2 = _hash_assistant("text2")
        assert h1 != h2

    @patch("memo.capture_core._extract_and_save")
    def test_extract_and_save_text_verbatim_fallback(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        """When extractor yields 0 candidates, verbatim save."""
        mem = MagicMock()
        mem.save.return_value = MagicMock(id="test-id", title="Verbatim save")
        cfg = MagicMock()
        cfg.state_dir = tmp_path

        # Mock _extract_and_save to return 0 candidates
        mock_extract.return_value = {
            "candidates": 0,
            "saved": [],
            "saved_titles": [],
            "skipped_dup": 0,
            "reconciled": 0,
            "skipped_quality": 0,
            "skipped_meta": 0,
            "skipped_batch_dup": 0,
            "uncertain": 0,
            "retyped": 0,
        }

        result = extract_and_save_text(
            mem, cfg, "explicit text to save", title="My Note"
        )
        assert result["status"] == "verbatim"
        assert result["saved"] == ["test-id"]

    @patch("memo.capture_core._extract_and_save")
    def test_extract_and_save_text_extracted(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        """When extractor yields candidates, return extracted status."""
        cfg = MagicMock()
        cfg.state_dir = tmp_path

        # Mock _extract_and_save to return 2 candidates extracted
        mock_extract.return_value = {
            "candidates": 2,
            "saved": ["id1", "id2"],
            "saved_titles": ["Fact 1", "Fact 2"],
            "skipped_dup": 0,
            "reconciled": 0,
            "skipped_quality": 0,
            "skipped_meta": 0,
            "skipped_batch_dup": 0,
            "uncertain": 0,
            "retyped": 0,
        }

        result = extract_and_save_text(MagicMock(), cfg, "text to extract")
        assert result["status"] == "extracted"
        assert result["candidates"] == 2
        assert len(result["saved"]) == 2


class TestIntegrationExtractAndSave:
    """Integration tests for core extraction pipeline."""

    def test_extract_and_save_quality_gate(self, tmp_path: Path) -> None:
        """Quality gate filters out low-specificity candidates."""
        mem = MagicMock()
        cfg = MagicMock()
        cfg.state_dir = tmp_path
        cfg.helper_model = "local"

        # Mock the helper chat to return a low-quality candidate
        helper = MagicMock()
        mem._ensure_chat.return_value = helper

        insights_json = json.dumps([
            {
                "title": "session note",  # Generic session narrative
                "type": "note",
                "body": "A",  # Too short for quality gate
                "tags": [],
            }
        ])
        helper.chat.return_value = {"message": {"content": insights_json}}

        from memo.capture_core import _extract_and_save

        with patch("memo.capture_core._passes_quality", return_value=False):
            result = _extract_and_save(
                mem, cfg, "user text", "assistant text", debug=False
            )
        # Candidate extracted by LLM but filtered by quality gate
        assert result["candidates"] == 1
        assert result["skipped_quality"] == 1
        assert len(result["saved"]) == 0

    def test_extract_and_save_meta_filter(self, tmp_path: Path) -> None:
        """Meta-commentary filter removes process narration."""
        mem = MagicMock()
        cfg = MagicMock()
        cfg.state_dir = tmp_path
        cfg.helper_model = "local"

        insights = [
            {
                "title": "voy a implementar",  # Meta-commentary title
                "type": "note",
                "body": "let me check the system",  # Pure narration
                "tags": [],
            }
        ]

        from memo.capture_core import _extract_and_save

        def capture_flag_enabled(name: str) -> bool:
            return name == "MEMO_CAPTURE_META_FILTER"

        with patch("memo.capture_core._passes_quality", return_value=False):
            with patch(
                "memo.capture_core._capture_flag_bool",
                side_effect=capture_flag_enabled,
            ):
                with patch("memo.capture_core.extract_insights", return_value=insights):
                    result = _extract_and_save(
                        mem, cfg, "user text", "assistant text", debug=False
                    )
        # Meta-commentary candidate is dropped
        assert result["candidates"] == 1
        assert result["skipped_meta"] == 1
