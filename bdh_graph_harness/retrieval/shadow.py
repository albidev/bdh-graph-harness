"""Privacy-safe telemetry for notes recovered only via learned edges."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def build_dynamic_shadow(query: str, routing: dict) -> dict:
    """Summarize dynamic-only retrieval without retaining query text."""
    dynamic_only = []
    for rank, detail in enumerate(routing.get("activation_details", []), start=1):
        if detail.get("matched_by") != "hebbian_edge":
            continue
        dynamic_only.append({
            "id": detail["id"],
            "rank": rank,
            "parent_id": detail.get("parent_id"),
            "weight": detail.get("hebbian_edge_weight", 0.0),
            "trust": detail.get("hebbian_edge_trust", 0.0),
            "score": detail.get("final_score", 0.0),
        })
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_fingerprint": sha256(query.encode()).hexdigest()[:16],
        "dynamic_only_count": len(dynamic_only),
        "dynamic_only": dynamic_only,
    }


def append_dynamic_shadow(vault_path: str | Path, shadow: dict) -> None:
    """Append one privacy-safe event to the vault-local JSONL telemetry log."""
    path = Path(vault_path) / ".bdh-hebbian-shadow.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(shadow, ensure_ascii=False, separators=(",", ":")) + "\n")
