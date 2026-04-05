"""
logger.py — Structured Logging
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — In-memory metrics tracking (replace with Prometheus in production)
# ─────────────────────────────────────────────────────────────────────────────

from collections import defaultdict
from datetime import datetime
from typing import Any

_metrics = {
    "total_tickets": 0,
    "resolved": 0,
    "escalated": 0,
    "by_intent": defaultdict(int),
    "by_urgency": defaultdict(int),
    "quality_scores": [],
    "iterations": [],
    "started_at": datetime.utcnow().isoformat(),
}


def track_ticket(
    intent: str,
    urgency: str,
    escalated: bool,
    quality_score: float,
    iterations: int,
) -> None:
    _metrics["total_tickets"] += 1
    if escalated:
        _metrics["escalated"] += 1
    else:
        _metrics["resolved"] += 1
    _metrics["by_intent"][intent] += 1
    _metrics["by_urgency"][urgency] += 1
    _metrics["quality_scores"].append(quality_score)
    _metrics["iterations"].append(iterations)


def get_metrics_summary() -> dict[str, Any]:
    total = _metrics["total_tickets"]
    scores = _metrics["quality_scores"]
    iters = _metrics["iterations"]

    return {
        "total_tickets": total,
        "resolved": _metrics["resolved"],
        "escalated": _metrics["escalated"],
        "resolution_rate": round(_metrics["resolved"] / total, 3) if total else 0,
        "escalation_rate": round(_metrics["escalated"] / total, 3) if total else 0,
        "avg_quality_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "avg_iterations": round(sum(iters) / len(iters), 2) if iters else 0,
        "by_intent": dict(_metrics["by_intent"]),
        "by_urgency": dict(_metrics["by_urgency"]),
        "system_started_at": _metrics["started_at"],
    }
