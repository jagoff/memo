from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime

from ._providers import _merge_candidates, default_resume_providers
from ._types import (
    MEMO_RESUME_REPORT_SCHEMA,
    ResumeCandidate,
    ResumeDiscoveryReport,
    ResumeProvider,
    ResumeProviderError,
    utc_now_iso,
)
from ._utils import (
    _STATUS_RANK,
    _clip,
    _normalize_agent_filter,
    _resolve_cwd,
    _sort_key,
    _with_run_status,
)


def discover_resume_candidates(
    *,
    agent: str = "all",
    cwd: str | None = None,
    include_all_cwd: bool = False,
    limit: int = 10,
    providers: Sequence[ResumeProvider] | None = None,
) -> ResumeDiscoveryReport:
    normalized_agent = _normalize_agent_filter(agent)
    cwd_value = _resolve_cwd(cwd or os.getcwd())
    effective_limit = max(1, int(limit))
    provider_list = list(providers) if providers is not None else default_resume_providers()
    candidates: list[ResumeCandidate] = []
    errors: list[ResumeProviderError] = []

    # Ask each provider for a broader page, then merge and trim globally.
    provider_limit = max(effective_limit * 4, effective_limit)
    for provider in provider_list:
        try:
            candidates.extend(
                provider.discover(
                    agent=normalized_agent,
                    cwd=cwd_value,
                    include_all_cwd=include_all_cwd,
                    limit=provider_limit,
                )
            )
        except Exception as exc:
            errors.append(ResumeProviderError(provider.name, _clip(str(exc), 240)))

    merged = _merge_candidates(candidates)
    now = datetime.now(UTC)
    finalized = [_with_run_status(item, now) for item in merged]
    # Active sessions first, then by recency. Tuple sort with reverse keeps both descending.
    finalized.sort(
        key=lambda item: (_STATUS_RANK.get(item.status, 0), _sort_key(item.updated_at)),
        reverse=True,
    )
    return ResumeDiscoveryReport(
        schema=MEMO_RESUME_REPORT_SCHEMA,
        generated_at=utc_now_iso(),
        agent=normalized_agent,
        cwd=cwd_value,
        limit=effective_limit,
        include_all_cwd=include_all_cwd,
        candidates=finalized[:effective_limit],
        provider_errors=errors,
    )
