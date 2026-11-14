"""Budget-aware packing of activated notes into a bounded context.

Retrieval returns a ranked set of notes; the context that carries them has a
fixed size. Filling it greedily lets one long note consume the whole budget and
silently drops every note ranked after it, while a fixed per-note cut wastes
the room short notes leave unused. This module shares one character budget
across the ranked notes instead:

* every admitted note is guaranteed a minimum excerpt, in ranking order, and
  the best-ranked note always gets in, shortened if the budget is that small;
* a note that cannot get that minimum is omitted, and packing continues with
  the notes after it rather than stopping;
* the remaining room is split in proportion to activation score, and whatever
  a short note does not need is handed on to the notes still wanting more;
* a shortened note is cut at a sentence or word boundary and marked.

Budgets are in characters, the unit every layer of the pipeline already uses.
No tokenizer is involved: the serving model varies per deployment, so a
character budget is the only figure that is exact here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TRUNCATION_MARKER = " […]"
DEFAULT_MIN_NOTE_CHARS = 400
DEFAULT_RETRIEVED_CONTEXT_BUDGET_CHARS = 8000
DEFAULT_LLM_CONTEXT_BUDGET_CHARS = 10000
MIN_CONTEXT_BUDGET_CHARS = 200
MAX_CONTEXT_BUDGET_CHARS = 64000

_SENTENCE_ENDINGS = (". ", "! ", "? ", ".\n", "!\n", "?\n", "\n")


@dataclass(frozen=True)
class PackItem:
    """One ranked note offered to the packer.

    ``overhead`` is the number of characters the caller spends around the body
    (heading, separator); it is charged against the budget for admitted notes.
    """

    key: str
    score: float
    text: str
    overhead: int = 0


@dataclass(frozen=True)
class PackedItem:
    """The share of the budget granted to one admitted note."""

    key: str
    text: str
    full_chars: int
    truncated: bool


def coerce_budget_chars(value, default: int) -> int:
    """Return a usable budget, falling back to ``default`` for unusable values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = default
    return max(MIN_CONTEXT_BUDGET_CHARS, min(int(value), MAX_CONTEXT_BUDGET_CHARS))


def truncate_at_boundary(text: str, limit: int, marker: str = TRUNCATION_MARKER) -> str:
    """Shorten ``text`` to at most ``limit`` characters, marker included.

    Prefers the last sentence end in the final quarter of the allowance, then
    the last whitespace, and only then a hard cut.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    room = limit - len(marker)
    if room <= 0:
        return text[:limit]
    head = text[:room]
    earliest = room - max(1, room // 4)
    cut = max(head.rfind(ending) + len(ending.rstrip()) for ending in _SENTENCE_ENDINGS)
    if cut <= earliest:
        cut = head.rfind(" ")
    if cut <= earliest:
        cut = room
    return head[:cut].rstrip() + marker


def pack_items(
    items: Sequence[PackItem],
    budget_chars: int,
    *,
    min_chars: int = DEFAULT_MIN_NOTE_CHARS,
) -> tuple[list[PackedItem], list[str]]:
    """Share ``budget_chars`` across ``items``, given in ranking order.

    Returns ``(packed, omitted_keys)``. ``packed`` keeps the input order. The
    sum of every packed body plus its ``overhead`` never exceeds the budget.
    """
    remaining = max(0, int(budget_chars))
    min_chars = max(1, int(min_chars))
    admitted: list[int] = []
    allocation: dict[int, int] = {}
    omitted: list[str] = []

    for index, item in enumerate(items):
        floor = min(len(item.text), min_chars)
        overhead = max(0, item.overhead)
        if not admitted and not omitted and remaining > overhead:
            # The best-ranked note always gets in: a budget too small for the
            # guaranteed excerpt carries a shorter one rather than a lesser note.
            floor = min(floor, remaining - overhead)
        cost = overhead + floor
        if cost > remaining:
            omitted.append(item.key)
            continue
        admitted.append(index)
        allocation[index] = floor
        remaining -= cost

    wanting = [index for index in admitted if len(items[index].text) > allocation[index]]
    while remaining > 0 and wanting:
        weights = {index: max(float(items[index].score or 0.0), 1e-6) for index in wanting}
        total_weight = sum(weights.values())
        pool = remaining
        for index in wanting:
            need = len(items[index].text) - allocation[index]
            grant = min(need, int(pool * weights[index] / total_weight))
            allocation[index] += grant
            remaining -= grant
        if remaining == pool:
            # Shares rounded down to nothing: hand the leftover characters to
            # the best-ranked note that still wants them.
            index = wanting[0]
            grant = min(remaining, len(items[index].text) - allocation[index])
            allocation[index] += grant
            remaining -= grant
        wanting = [index for index in wanting if len(items[index].text) > allocation[index]]

    packed = []
    for index in admitted:
        item = items[index]
        body = truncate_at_boundary(item.text, allocation[index])
        packed.append(PackedItem(
            key=item.key,
            text=body,
            full_chars=len(item.text),
            truncated=len(body) < len(item.text),
        ))
    return packed, omitted


def build_retrieved_context(
    activated_notes: Sequence[dict],
    nodes: dict,
    budget_chars: int,
    *,
    min_note_chars: int = DEFAULT_MIN_NOTE_CHARS,
) -> tuple[str, dict]:
    """Render ranked activated notes as ``## Title`` sections within a budget.

    ``activated_notes`` is the ranked API payload (``id``, ``title``,
    ``score``). Returns the context string and a metadata dict describing what
    each note received; the metadata carries ids and sizes only, never text.
    """
    separator = "\n\n"
    items, headings = [], {}
    for note in activated_notes:
        node = nodes.get(note.get("id"))
        text = node.get("text", "") if isinstance(node, dict) else ""
        if not isinstance(text, str) or not text.strip():
            continue
        heading = f"## {note.get('title', note['id'])}\n"
        headings[note["id"]] = heading
        items.append(PackItem(
            key=note["id"],
            score=float(note.get("score") or 0.0),
            text=text.strip(),
            overhead=len(heading) + len(separator),
        ))

    packed, omitted = pack_items(items, budget_chars, min_chars=min_note_chars)
    context = separator.join(f"{headings[entry.key]}{entry.text}" for entry in packed)
    meta = {
        "budget_chars": int(budget_chars),
        "used_chars": len(context),
        "notes": [
            {
                "id": entry.key,
                "chars": len(entry.text),
                "full_chars": entry.full_chars,
                "truncated": entry.truncated,
            }
            for entry in packed
        ],
        "omitted": omitted,
    }
    return context, meta
