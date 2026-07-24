"""Backward-compatible aliases for the OpenAI-compatible helpers.

New code should import from :mod:`openai_compatible`; this module remains so
external integrations using the old import path do not break.
"""

from bdh_graph_harness.llm.openai_compatible import (
    build_openai_compatible_payload as build_openrouter_payload,
    parse_openai_compatible_response as parse_openrouter_response,
    parse_openai_compatible_stream_token as parse_openrouter_stream_token,
)

__all__ = [
    'build_openrouter_payload',
    'parse_openrouter_response',
    'parse_openrouter_stream_token',
]
