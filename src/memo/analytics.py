"""Memory analytics dashboard — metrics and visualizations.

Provides:
- Dashboard web UI with corpus metrics
- Growth charts over time
- Distribution by type/tags
- Access patterns (heatmaps)
- Word cloud of most frequent entities
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any


def _ensure_output_parent(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class CorpusMetrics:
    """Metrics about the memory corpus."""

    total_memories: int
    total_entities: int
    type_distribution: dict[str, int]
    tag_frequency: dict[str, int]
    entity_frequency: dict[str, int]
    growth_rate: float  # memories per day
    average_access_count: float


@dataclass
class GrowthData:
    """Growth data over time."""

    dates: list[str]
    counts: list[int]


@dataclass
class AccessPattern:
    """Access pattern data."""

    day_of_week: dict[str, int]  # Monday-Sunday
    hour_of_day: dict[int, int]  # 0-23


class AnalyticsEngine:
    """Computes analytics metrics for the memory corpus.

    Args:
        memory: The Memory instance to analyze.
    """

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def compute_corpus_metrics(self) -> CorpusMetrics:
        """Compute comprehensive corpus metrics.

        Returns:
            CorpusMetrics with all computed metrics.
        """
        # Get all memories
        memories = self.memory.list(limit=10000)

        total_memories = len(memories)

        # Type distribution
        type_counter = Counter(m.type for m in memories)
        type_distribution = dict(type_counter)

        # Tag frequency
        tag_counter: Counter[str] = Counter()
        for m in memories:
            tag_counter.update(m.tags)
        tag_frequency = dict(tag_counter.most_common(50))

        # Entity frequency
        entity_counter: Counter[str] = Counter()
        for m in memories:
            entities = self.memory.graph.get_entity_mentions(m.id)
            for e in entities:
                entity_counter[e.name] += 1
        entity_frequency = dict(entity_counter.most_common(50))

        # Total entities
        total_entities = self.memory.graph.count_entities()

        # Growth rate (memories per day)
        growth_rate = 0.0
        if total_memories > 1:
            first_raw = memories[-1].updated
            last_raw = memories[0].updated
            if first_raw and last_raw:
                first_date = datetime.fromisoformat(first_raw.replace("Z", "+00:00"))
                last_date = datetime.fromisoformat(last_raw.replace("Z", "+00:00"))
                days = max(1.0, (last_date - first_date).total_seconds() / 86400)
                growth_rate = total_memories / days

        # Average access count
        total_access = sum(self.memory.lifecycle.get_access_count(m.id) for m in memories)
        average_access_count = total_access / total_memories if total_memories > 0 else 0.0

        return CorpusMetrics(
            total_memories=total_memories,
            total_entities=total_entities,
            type_distribution=type_distribution,
            tag_frequency=tag_frequency,
            entity_frequency=entity_frequency,
            growth_rate=growth_rate,
            average_access_count=average_access_count,
        )

    def compute_growth_data(self, days: int = 30) -> GrowthData:
        """Compute growth data over time.

        Args:
            days: Number of days to analyze.

        Returns:
            GrowthData with dates and counts.
        """
        memories = self.memory.list(limit=10000)

        # Group by date
        date_counts: Counter[str] = Counter()
        for m in memories:
            date_str = m.updated[:10]  # YYYY-MM-DD
            date_counts[date_str] += 1

        # Get last N days
        sorted_dates = sorted(date_counts.keys())[-days:]

        return GrowthData(
            dates=sorted_dates,
            counts=[date_counts[d] for d in sorted_dates],
        )

    def compute_access_patterns(self) -> AccessPattern:
        """Compute access patterns by day and hour.

        Returns:
            AccessPattern with day and hour distributions.
        """
        # This would analyze history store for access patterns
        # For now, return placeholder data
        return AccessPattern(
            day_of_week={
                "Monday": 0,
                "Tuesday": 0,
                "Wednesday": 0,
                "Thursday": 0,
                "Friday": 0,
                "Saturday": 0,
                "Sunday": 0,
            },
            hour_of_day={i: 0 for i in range(24)},
        )

    def export_metrics_json(self, output_path: Path) -> None:
        """Export metrics to JSON.

        Args:
            output_path: Path to write JSON file.
        """
        metrics = self.compute_corpus_metrics()
        growth = self.compute_growth_data()
        access = self.compute_access_patterns()

        data = {
            "metrics": metrics.__dict__,
            "growth": growth.__dict__,
            "access": access.__dict__,
            "exported_at": datetime.now(UTC).isoformat(),
        }

        _ensure_output_parent(output_path)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_metrics_csv(self, output_path: Path) -> None:
        """Export metrics to CSV.

        Args:
            output_path: Path to write CSV file.
        """
        import csv

        metrics = self.compute_corpus_metrics()

        _ensure_output_parent(output_path)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])

            writer.writerow(["Total Memories", metrics.total_memories])
            writer.writerow(["Total Entities", metrics.total_entities])
            writer.writerow(["Growth Rate", metrics.growth_rate])
            writer.writerow(["Average Access Count", metrics.average_access_count])

            writer.writerow([])
            writer.writerow(["Type", "Count"])
            for t, c in metrics.type_distribution.items():
                writer.writerow([t, c])

            writer.writerow([])
            writer.writerow(["Tag", "Count"])
            for t, c in metrics.tag_frequency.items():
                writer.writerow([t, c])

            writer.writerow([])
            writer.writerow(["Entity", "Count"])
            for e, c in metrics.entity_frequency.items():
                writer.writerow([e, c])


class Dashboard:
    """Generates dashboard visualizations.

    Args:
        analytics: The AnalyticsEngine instance.
    """

    def __init__(self, analytics: AnalyticsEngine) -> None:
        self.analytics = analytics

    def generate_summary(self) -> str:
        """Generate a text summary of the dashboard.

        Returns:
            Formatted summary string.
        """
        metrics = self.analytics.compute_corpus_metrics()

        lines = [
            "=== Memory Analytics Dashboard ===",
            "",
            f"Total Memories: {metrics.total_memories}",
            f"Total Entities: {metrics.total_entities}",
            f"Growth Rate: {metrics.growth_rate:.2f} memories/day",
            f"Average Access Count: {metrics.average_access_count:.2f}",
            "",
            "Type Distribution:",
        ]

        for t, c in metrics.type_distribution.items():
            lines.append(f"  {t}: {c}")

        lines.append("")
        lines.append("Top 10 Tags:")

        for i, (t, c) in enumerate(list(metrics.tag_frequency.items())[:10], 1):
            lines.append(f"  {i}. {t}: {c}")

        lines.append("")
        lines.append("Top 10 Entities:")

        for i, (e, c) in enumerate(list(metrics.entity_frequency.items())[:10], 1):
            lines.append(f"  {i}. {e}: {c}")

        return "\n".join(lines)

    def generate_html_dashboard(self, output_path: Path) -> None:
        """Generate an HTML dashboard.

        Args:
            output_path: Path to write HTML file.
        """
        metrics = self.analytics.compute_corpus_metrics()
        growth = self.analytics.compute_growth_data()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Analytics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ font-size: 24px; font-weight: bold; }}
        .section {{ margin: 30px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Memory Analytics Dashboard</h1>

    <div class="section">
        <h2>Overview</h2>
        <p>Total Memories: <span class="metric">{metrics.total_memories}</span></p>
        <p>Total Entities: <span class="metric">{metrics.total_entities}</span></p>
        <p>Growth Rate: <span class="metric">{metrics.growth_rate:.2f}</span> memories/day</p>
        <p>Average Access Count: <span class="metric">{metrics.average_access_count:.2f}</span></p>
    </div>

    <div class="section">
        <h2>Type Distribution</h2>
        <table>
            <tr><th>Type</th><th>Count</th></tr>
            {"".join(f"<tr><td>{escape(t)}</td><td>{c}</td></tr>" for t, c in metrics.type_distribution.items())}
        </table>
    </div>

    <div class="section">
        <h2>Top Tags</h2>
        <table>
            <tr><th>Tag</th><th>Count</th></tr>
            {"".join(f"<tr><td>{escape(t)}</td><td>{c}</td></tr>" for t, c in list(metrics.tag_frequency.items())[:20])}
        </table>
    </div>

    <div class="section">
        <h2>Top Entities</h2>
        <table>
            <tr><th>Entity</th><th>Count</th></tr>
            {"".join(f"<tr><td>{escape(e)}</td><td>{c}</td></tr>" for e, c in list(metrics.entity_frequency.items())[:20])}
        </table>
    </div>

    <div class="section">
        <h2>Growth (Last 30 Days)</h2>
        <table>
            <tr><th>Date</th><th>Count</th></tr>
            {"".join(f"<tr><td>{d}</td><td>{c}</td></tr>" for d, c in zip(growth.dates, growth.counts, strict=False))}
        </table>
    </div>

    <p>Generated: {datetime.now(UTC).isoformat()}</p>
</body>
</html>
"""

        _ensure_output_parent(output_path)
        output_path.write_text(html, encoding="utf-8")


__all__ = [
    "AccessPattern",
    "AnalyticsEngine",
    "CorpusMetrics",
    "Dashboard",
    "GrowthData",
]
