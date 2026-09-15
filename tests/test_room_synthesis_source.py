"""Invariants for the room_synthesis Curate-gated source.

Room synthesis is the hosted-group-room counterpart of session synthesis. It
must be registered as a first-class BDH source and pass through the same
pre-write Curate gate: no direct Hebbian plasticity, no direct neurogenesis,
candidates staged for review, and correlation carried on the candidate.

Regression for the gap that made room synthesis undeployable: the source was
unregistered, so ``source_policy`` raised ``ValueError`` at the policy boundary
and the whole request failed. Unknown sources are rejected by design, so a new
synthesis source must be registered explicitly — it can never be assumed.
"""
import pytest

from bdh_graph_harness.memory.source_policy import (
    CURATE_GATED_SOURCES,
    allowed_sources,
    get_frequency_increment,
    get_source_policy,
    is_curate_gated,
    use_user_prompt_for_retrieval,
)


class TestRoomSynthesisSourceRegistration:
    def test_room_synthesis_is_a_registered_source(self):
        assert "room_synthesis" in allowed_sources()

    def test_room_synthesis_policy_is_resolvable(self):
        """An unregistered source raises here; that is the original failure."""
        policy = get_source_policy("room_synthesis")
        assert policy is not None
        assert policy.name == "room_synthesis"
        assert policy.allow_neurogenesis is True

    def test_room_synthesis_frequency_increment_resolves(self):
        """The policy boundary rejects unknown sources — this must not raise."""
        assert get_frequency_increment("room_synthesis") == pytest.approx(0.2)

    def test_room_synthesis_uses_the_transcript_for_retrieval(self):
        """The transcript is the evidence; the query is a generic label.

        Without this the attention pass would run on an unrelated string, the
        issue #16 defect.
        """
        assert use_user_prompt_for_retrieval("room_synthesis") is True

    def test_room_synthesis_is_damped_like_session_synthesis(self):
        room = get_source_policy("room_synthesis")
        session = get_source_policy("session_synthesis")
        assert room is not None and session is not None
        assert room.frequency_increment == session.frequency_increment
        assert room.provenance_label == "room_synthesis"


class TestCurateGateMembership:
    def test_both_synthesis_sources_are_curate_gated(self):
        assert CURATE_GATED_SOURCES == {"session_synthesis", "room_synthesis"}
        assert is_curate_gated("session_synthesis") is True
        assert is_curate_gated("room_synthesis") is True

    def test_non_synthesis_sources_are_not_gated(self):
        for source in ("user_query", "cron", "automatic_retrieval",
                       "assistant_response", "nightly_semantic_consolidation"):
            assert is_curate_gated(source) is False

    def test_unknown_source_is_not_gated_rather_than_assumed(self):
        assert is_curate_gated("something_new") is False
        assert is_curate_gated(None) is False


class TestRoomSynthesisStagingContract:
    """Staging must label candidates with the source that produced them.

    The candidate ``source`` feeds the Curate correlation check, so a room
    candidate stamped as session_synthesis would be rejected at apply time.
    """

    def test_staged_candidate_carries_the_room_source(self, tmp_path, monkeypatch):
        from bdh_graph_harness.memory import session_synthesis_staging as staging

        monkeypatch.setattr(
            staging, "_extract_concepts_safely",
            lambda *a, **k: [{
                "title": "Room Concept",
                "definition": "A durable concept agreed in a room.",
                "confidence": "high",
            }],
        )
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-room-1",
            vault_id="vault-a",
            session_id="room-1",
            transcript_sha256="a" * 64,
            response_text="irrelevant",
            source="room_synthesis",
            config={"session_synthesis_staging_max_concepts": 1},
        )

        assert result["count"] == 1
        candidate = result["candidates"][0]
        assert candidate["source"] == "room_synthesis"
        assert candidate["session_id"] == "room-1"
        assert candidate["status"] == "pending_review"

    def test_curate_audit_row_is_opened_for_the_room_candidate(self, tmp_path, monkeypatch):
        from bdh_graph_harness.memory import session_synthesis_staging as staging
        from bdh_graph_harness.memory.curate_audit import latest_curate_state

        monkeypatch.setattr(
            staging, "_extract_concepts_safely",
            lambda *a, **k: [{
                "title": "Room Concept",
                "definition": "A durable concept agreed in a room.",
                "confidence": "high",
            }],
        )
        result = staging.stage_session_synthesis_candidates(
            str(tmp_path),
            synthesis_id="syn-room-2",
            vault_id="vault-a",
            session_id="room-2",
            transcript_sha256="b" * 64,
            response_text="irrelevant",
            source="room_synthesis",
            config={"session_synthesis_staging_max_concepts": 1},
        )

        cid = result["candidates"][0]["candidate_id"]
        entry = latest_curate_state(str(tmp_path), cid)
        assert entry is not None
        assert entry.state == "pending_review"
        assert entry.extra.get("source") == "room_synthesis"
