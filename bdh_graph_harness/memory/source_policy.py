"""Explicit source policy for Hebbian plasticity and retrieval routing.

Every ``source`` string accepted by ``/api/query`` must be registered here
with an explicit ``frequency_increment`` and provenance label.  Unknown
sources are *rejected* at the policy boundary — they no longer silently
fall through to full-strength interactive learning.

Design goals (GitHub issue #16)
-------------------------------
1. **No silent fall-through** — unrecognised source strings raise
   ``ValueError`` instead of defaulting to ``frequency_increment=1.0``.
2. **Explicit dampening tiers** — interactive queries get full strength,
   derived/secondary signals get reduced increments.
3. **Provenance** — every Hebbian update carries a ``source_policy``
   record so downstream consumers can trace where a weight change came
   from.
4. **Retrieval routing** — ``session_synthesis`` uses the session
   ``user_prompt`` for attention instead of a fixed generic query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Source policy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePolicy:
    """Immutable descriptor for one ``source`` value.

    Attributes
    ----------
    name:
        Canonical source identifier (e.g. ``"user_query"``).
    frequency_increment:
        Multiplier applied to the raw co-activation score product before
        adding to synapse frequency.  ``1.0`` = full interactive strength.
    use_user_prompt_for_retrieval:
        When ``True`` and ``user_prompt`` is provided in the request body,
        the attention/plasticity pass uses ``user_prompt`` instead of
        ``query`` as its retrieval signal.  This is essential for
        ``session_synthesis`` where the bridge places the transcript in
        ``user_prompt`` while ``query`` is a fixed generic label.
    provenance_label:
        Human-readable provenance tag stored on each updated synapse.
    allow_neurogenesis:
        Whether neurogenesis should run for this source.  ``cron`` and
        similar batch sources typically disable it.
    """

    name: str
    frequency_increment: float = 1.0
    use_user_prompt_for_retrieval: bool = False
    provenance_label: str = ""
    allow_neurogenesis: bool = True


# ---------------------------------------------------------------------------
# Registry — the single source of truth
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, SourcePolicy] = {}


def register_source(
    name: str,
    *,
    frequency_increment: float = 1.0,
    use_user_prompt_for_retrieval: bool = False,
    provenance_label: str = "",
    allow_neurogenesis: bool = True,
) -> SourcePolicy:
    """Register a source policy (called at module load time)."""
    policy = SourcePolicy(
        name=name,
        frequency_increment=frequency_increment,
        use_user_prompt_for_retrieval=use_user_prompt_for_retrieval,
        provenance_label=provenance_label or name,
        allow_neurogenesis=allow_neurogenesis,
    )
    _REGISTRY[name] = policy
    return policy


# Built-in sources -----------------------------------------------------------

register_source(
    "user_query",
    frequency_increment=1.0,
    provenance_label="interactive_query",
    allow_neurogenesis=True,
)

register_source(
    "assistant_response",
    frequency_increment=0.3,
    provenance_label="assistant_response",
    allow_neurogenesis=True,
)

register_source(
    "nightly_semantic_consolidation",
    frequency_increment=0.3,
    provenance_label="semantic_consolidation",
    allow_neurogenesis=True,
)

register_source(
    "session_synthesis",
    frequency_increment=0.2,
    use_user_prompt_for_retrieval=True,
    provenance_label="session_synthesis",
    allow_neurogenesis=True,
)

register_source(
    "cron",
    frequency_increment=0.3,
    provenance_label="cron",
    allow_neurogenesis=False,
)

register_source(
    "automatic_retrieval",
    frequency_increment=1.0,
    provenance_label="automatic_retrieval",
    allow_neurogenesis=True,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Sources that existed before issue #16 with their old hardcoded values.
# The explicit registry above supersedes these, but we keep the set for
# documentation / migration checks.
_LEGACY_SOURCES: FrozenSet[str] = frozenset({
    "assistant_response",
    "nightly_semantic_consolidation",
})


def get_frequency_increment(source: str | None) -> float:
    """Return the Hebbian frequency increment for *source*.

    Raises ``ValueError`` for unrecognised sources — this is the
    anti-fall-through guard introduced by issue #16.
    """
    if source is None:
        # ``None`` means no explicit source was provided; treat as
        # interactive (full strength).  This preserves the pre-issue-16
        # default for callers that omit ``source``.
        return 1.0
    policy = _REGISTRY.get(source)
    if policy is None:
        raise ValueError(
            f"Unknown source {source!r}. "
            f"Recognised sources: {sorted(_REGISTRY)}. "
            f"Register it in bdh_graph_harness.memory.source_policy."
        )
    return policy.frequency_increment


def get_source_policy(source: str | None) -> SourcePolicy | None:
    """Return the full policy for *source*, or ``None`` if source is None."""
    if source is None:
        return None
    return _REGISTRY.get(source)


def use_user_prompt_for_retrieval(source: str | None) -> bool:
    """Whether the attention pass should use ``user_prompt`` as the
    retrieval query when *source* is active."""
    if source is None:
        return False
    policy = _REGISTRY.get(source)
    if policy is None:
        raise ValueError(f"Unknown source {source!r}")
    return policy.use_user_prompt_for_retrieval


def allowed_sources() -> list[str]:
    """Return sorted list of all registered source names."""
    return sorted(_REGISTRY)
