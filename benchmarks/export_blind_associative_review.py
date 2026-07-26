#!/usr/bin/env python3
"""Export a blind, local-only qualitative comparison of associative candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("candidate_a", "candidate_b", "candidate_c")


def note_preview(vault: Path, note_id: str) -> dict[str, str]:
    path = vault / f"{note_id}.md"
    if not path.exists():
        return {"note_id": note_id, "title": note_id.rsplit("/", 1)[-1], "excerpt": "[note unavailable]"}
    text = path.read_text(encoding="utf-8", errors="replace")
    title = re.search(r"(?m)^title:\s*[\"']?(.+?)[\"']?\s*$", text)
    if not title:
        title = re.search(r"(?m)^#\s+(.+)$", text)
    body = re.sub(r"\A---.*?---\s*", "", text, flags=re.S)
    body = re.sub(r"(?m)^#.*$", "", body)
    excerpt = re.sub(r"\s+", " ", body).strip()[:500]
    return {
        "note_id": note_id,
        "title": title.group(1).strip() if title else note_id.rsplit("/", 1)[-1],
        "excerpt": excerpt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.artifact.read_text())
    cases, answer_key = [], {}
    for index, record in enumerate(source["records"], start=1):
        lanes = {
            "hebbian": [item["id"] for item in record.get("associative_context", [])],
            "static_neighbor": record.get("static_neighbor_control", []),
            "embedding_neighbor": record.get("embedding_neighbor_control", []),
        }
        if not any(lanes.values()):
            continue
        order = list(lanes)
        random.Random(hashlib.sha256(record["query"].encode()).digest()).shuffle(order)
        assignment = dict(zip(LABELS, order))
        cases.append({
            "case_id": f"assoc-{index:03d}",
            "query": record["query"],
            "candidates": {
                label: [note_preview(args.vault, note_id) for note_id in lanes[lane]]
                for label, lane in assignment.items()
            },
            "labels": {label: None for label in LABELS},
            "allowed_labels": ["useful", "redundant", "noise", "misleading", "no_candidate"],
        })
        answer_key[f"assoc-{index:03d}"] = assignment

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pack = args.output_dir / f"blind-associative-review-{stamp}.json"
    key = args.output_dir / f"blind-associative-review-{stamp}.key.json"
    pack.write_text(json.dumps({"protocol": "blind-associative-comparison-v1", "cases": cases}, ensure_ascii=False, indent=2) + "\n")
    key.write_text(json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n")
    print(pack)
    print(key)
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
