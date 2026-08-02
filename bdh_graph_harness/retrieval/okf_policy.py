"""OKF-aware retrieval policy.

This module is deliberately a read-only policy layer. It interprets document
metadata for ranking, but never writes OKF metadata or mutates BDH runtime state
(including Hebbian synapses).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse

from bdh_graph_harness.config import CONFIG


_LOCAL_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_OFF_MODES = {False, None, "", "off", 0}


def is_okf_retrieval_policy_enabled() -> bool:
    """Return whether OKF metadata may affect retrieval ranking.

    ``okf_mode`` is the compatibility gate. The separate policy flag lets a
    deployment keep OKF parsing/export enabled while temporarily disabling
    trust-aware ranking during an experiment.
    """
    mode = CONFIG.get("okf_mode", False)
    return bool(CONFIG.get("okf_retrieval_policy_enabled", True)) and mode not in _OFF_MODES


def _metadata_for_node(node: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        return {}
    metadata = node.get("okf")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _coerce_datetime(value: Any, *, default: datetime | None = None) -> datetime:
    if value is None:
        result = default or datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            result = default or datetime.now(timezone.utc)

    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _parse_stale_after(value: Any) -> tuple[str | None, date | None]:
    if value is None or isinstance(value, bool):
        return None, None
    if isinstance(value, datetime):
        return value.date().isoformat(), value.date()
    if isinstance(value, date):
        return value.isoformat(), value

    text = str(value).strip()
    if not text:
        return None, None
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        return text, None
    return parsed.isoformat(), parsed


def _verified_state(value: Any) -> tuple[str, int]:
    """Return a public state label and count without exposing actor identities."""
    if value is None:
        return "unknown", 0
    if isinstance(value, bool):
        return ("verified", 1) if value else ("unverified", 0)
    if isinstance(value, Mapping):
        return ("verified", 1) if value else ("unknown", 0)
    if isinstance(value, (list, tuple)):
        valid_entries = sum(1 for item in value if isinstance(item, Mapping) and item)
        return ("verified", valid_entries) if valid_entries else ("unknown", 0)
    return "unknown", 0


def _source_type(resource: Any) -> str:
    text = str(resource or "").strip()
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        return "url"
    if text.startswith("redacted://"):
        return "redacted"
    if text.startswith(("/", "~")) or _LOCAL_PATH_RE.match(text):
        return "local"
    if text.startswith(("./", "../")) or "/" in text or text.endswith(".md"):
        return "bundle"
    return "other"


def _provenance_signal(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw_sources = metadata.get("sources")
    if isinstance(raw_sources, Mapping):
        sources = [raw_sources]
    elif isinstance(raw_sources, (list, tuple)):
        sources = list(raw_sources)
    else:
        sources = []

    source_types: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        resource = source.get("resource")
        if resource is None or not str(resource).strip():
            continue
        kind = _source_type(resource)
        if kind not in source_types:
            source_types.append(kind)

    source_count = sum(
        1
        for source in sources
        if isinstance(source, Mapping)
        and source.get("resource") is not None
        and str(source.get("resource")).strip()
    )
    generated = metadata.get("generated")
    generated_by_present = (
        isinstance(generated, Mapping)
        and bool(str(generated.get("by", "")).strip())
    )
    generated_at_present = (
        isinstance(generated, Mapping)
        and bool(str(generated.get("at", "")).strip())
    )
    has_provenance = bool(source_count or generated_by_present or generated_at_present)
    factor = (
        float(CONFIG.get("okf_policy_provenance_bonus", 1.03))
        if source_count
        else 1.0
    )
    return {
        "source_count": source_count,
        "source_types": source_types,
        "generated_by_present": generated_by_present,
        "generated_at_present": generated_at_present,
        "has_provenance": has_provenance,
        "factor": factor,
    }


def evaluate_okf_metadata(
    node: Mapping[str, Any] | None,
    *,
    now: datetime | date | str | None = None,
) -> dict[str, Any]:
    """Evaluate OKF trust, freshness, and provenance signals for one node.

    The result intentionally preserves the dimensions separately. ``multiplier``
    is only the deterministic ranking application of those signals; callers can
    inspect the component decisions in routing metadata.
    """
    metadata = _metadata_for_node(node)
    if not metadata:
        return {
            "applied": False,
            "status": {"value": "unknown", "factor": 1.0},
            "verified": {"state": "unknown", "count": 0, "factor": 1.0},
            "freshness": {"state": "unknown", "stale_after": None, "factor": 1.0},
            "provenance": {
                "source_count": 0,
                "source_types": [],
                "generated_by_present": False,
                "generated_at_present": False,
                "has_provenance": False,
                "factor": 1.0,
            },
            "multiplier": 1.0,
            "reasons": [],
        }

    status_value = str(metadata.get("status", "unknown") or "unknown").strip().lower()
    status_factors = {
        "draft": float(CONFIG.get("okf_policy_draft_multiplier", 0.85)),
        "deprecated": float(CONFIG.get("okf_policy_deprecated_multiplier", 0.50)),
    }
    status_factor = status_factors.get(status_value, 1.0)
    status = {"value": status_value, "factor": status_factor}

    verified_value, verified_count = _verified_state(metadata.get("verified"))
    verified_factors = {
        "verified": float(CONFIG.get("okf_policy_verified_bonus", 1.08)),
        "unverified": float(CONFIG.get("okf_policy_unverified_penalty", 0.95)),
    }
    verified_factor = verified_factors.get(verified_value, 1.0)
    verified = {
        "state": verified_value,
        "count": verified_count,
        "factor": verified_factor,
    }

    stale_after, stale_date = _parse_stale_after(metadata.get("stale_after"))
    if stale_date is None:
        freshness_state = "invalid" if stale_after else "unknown"
        freshness_factor = 1.0
    else:
        current_date = _coerce_datetime(now).date()
        freshness_state = "stale" if current_date > stale_date else "fresh"
        freshness_factor = (
            float(CONFIG.get("okf_policy_stale_multiplier", 0.60))
            if freshness_state == "stale"
            else 1.0
        )
    freshness = {
        "state": freshness_state,
        "stale_after": stale_after,
        "factor": freshness_factor,
    }

    provenance = _provenance_signal(metadata)
    multiplier = status_factor * verified_factor * freshness_factor * provenance["factor"]
    minimum = float(CONFIG.get("okf_policy_min_multiplier", 0.35))
    maximum = float(CONFIG.get("okf_policy_max_multiplier", 1.20))
    multiplier = min(maximum, max(minimum, multiplier))

    reasons: list[str] = []
    if status_value in {"draft", "deprecated"}:
        reasons.append(f"status:{status_value}")
    if verified_value in {"verified", "unverified"}:
        reasons.append(f"verification:{verified_value}")
    if freshness_state == "stale":
        reasons.append("freshness:stale")
    if provenance["source_count"]:
        reasons.append(f"provenance:{provenance['source_count']}")
    if provenance["generated_by_present"] or provenance["generated_at_present"]:
        reasons.append("provenance:generated")

    return {
        "applied": True,
        "status": status,
        "verified": verified,
        "freshness": freshness,
        "provenance": provenance,
        "multiplier": round(multiplier, 6),
        "reasons": reasons,
    }


def apply_okf_retrieval_policy(
    scores: Mapping[str, float],
    nodes: Mapping[str, Mapping[str, Any]],
    *,
    now: datetime | date | str | None = None,
    enabled: bool | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Apply OKF metadata as a deterministic, non-mutating ranking adjustment."""
    if enabled is None:
        enabled = is_okf_retrieval_policy_enabled()
    if not enabled:
        return dict(scores), {}

    adjusted: dict[str, float] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for note_id, score in scores.items():
        decision = evaluate_okf_metadata(nodes.get(note_id), now=now)
        decisions[note_id] = decision
        adjusted[note_id] = float(score) * float(decision["multiplier"])
    return adjusted, decisions


__all__ = [
    "apply_okf_retrieval_policy",
    "evaluate_okf_metadata",
    "is_okf_retrieval_policy_enabled",
]
