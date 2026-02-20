from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_latest_eval_report(report_dir: str = "evals/reports") -> tuple[str | None, dict | None]:
    """
    Find the latest eval report on disk.
    Preference:
      1) latest.json in report dirs
      2) most recent timestamp in filename (YYYYMMDD_HHMMSS)
      3) newest modified time
    """
    candidate_dirs = [Path(report_dir), Path("evals/reports")]
    seen = set()
    dirs = []
    for directory in candidate_dirs:
        key = str(directory.resolve()) if directory.exists() else str(directory)
        if key not in seen:
            seen.add(key)
            dirs.append(directory)

    report_files = []
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            report_files.append(path)

    if not report_files:
        return None, None

    ts_pattern = re.compile(r"(\d{8}_\d{6})")

    def sort_key(path: Path):
        is_latest = 1 if path.name.lower() == "latest.json" else 0
        match = ts_pattern.search(path.name)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                return (3, is_latest, dt.timestamp(), path.stat().st_mtime)
            except Exception:
                pass
        return (2, is_latest, path.stat().st_mtime, path.stat().st_mtime)

    sorted_candidates = sorted(report_files, key=sort_key, reverse=True)
    fallback: tuple[str | None, dict | None] = (None, None)
    for candidate in sorted_candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if fallback == (None, None):
            fallback = (str(candidate), payload)

        summary = payload.get("summary") if isinstance(payload, dict) else None
        top = payload if isinstance(payload, dict) else {}
        metric_keys = {
            "decision_accuracy",
            "refusal_precision",
            "answer_citation_compliance",
            "recall_at_k",
            "recall@k",
            "refusal_reason_accuracy",
            "empty_retrieval_rate",
        }
        if isinstance(summary, dict) and any(k in summary for k in metric_keys):
            return str(candidate), payload
        if isinstance(top, dict) and any(k in top for k in metric_keys):
            return str(candidate), payload

    return fallback


def eval_value(report: dict, key: str):
    summary = report.get("summary") if isinstance(report, dict) else None
    if isinstance(summary, dict) and key in summary:
        return summary.get(key)
    if isinstance(report, dict):
        return report.get(key)
    return None


def fmt_metric_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "—"


def fmt_int(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "—"


def get_last_run_text(report: dict, source_path: str) -> str:
    timestamp = None
    meta = report.get("meta") if isinstance(report, dict) else None
    if isinstance(meta, dict):
        timestamp = meta.get("timestamp")
    timestamp = timestamp or eval_value(report, "timestamp")
    if isinstance(timestamp, str) and timestamp:
        if re.match(r"^\d{8}_\d{6}$", timestamp):
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                return timestamp
        return timestamp

    try:
        dt = datetime.fromtimestamp(Path(source_path).stat().st_mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "unknown"
