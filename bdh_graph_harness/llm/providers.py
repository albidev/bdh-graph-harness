"""LLM provider dispatch — builds payloads, parses responses, and drives LLM calls.

Dispatcher functions (_build_llm_payload, _parse_llm_response, _parse_llm_stream_token)
delegate to ollama.py or openrouter.py based on CONFIG['llm_provider'].
"""

import json
import re

from bdh_graph_harness.config import retry_with_backoff
import bdh_graph_harness.config as _config
from bdh_graph_harness.llm.prompt import build_messages, format_context
from bdh_graph_harness.llm.ollama import build_ollama_payload, parse_ollama_response
from bdh_graph_harness.llm.openai_compatible import (
    build_openai_compatible_payload,
    parse_openai_compatible_response,
    parse_openai_compatible_stream_token,
)


OPENAI_COMPATIBLE_PROVIDERS = frozenset({'openrouter', 'ollama-cloud'})


def uses_openai_compatible_api(provider=None):
    """Return whether *provider* speaks the Chat Completions contract."""
    provider = provider or _config.CONFIG.get('llm_provider', 'ollama')
    return provider in OPENAI_COMPATIBLE_PROVIDERS


def _build_llm_payload(query, active_notes, nodes, stream=False):
    """Build request payload + headers for the configured LLM provider.

    Returns (data_bytes, headers_dict).
    """
    messages = build_messages(query, active_notes, nodes)
    provider = _config.CONFIG.get('llm_provider', 'ollama')

    if uses_openai_compatible_api(provider):
        return build_openai_compatible_payload(messages, stream, _config.CONFIG)
    else:
        return build_ollama_payload(messages, stream, _config.CONFIG)


def _parse_llm_response(result, provider='ollama'):
    """Parse LLM response from either provider format."""
    if uses_openai_compatible_api(provider):
        return parse_openai_compatible_response(result)
    else:
        return parse_ollama_response(result)


def _parse_llm_stream_token(obj, provider='ollama'):
    """Parse a single streaming chunk from either provider."""
    if uses_openai_compatible_api(provider):
        return parse_openai_compatible_stream_token(obj)
    else:
        # Ollama: message.content
        if obj.get('done', False):
            return None
        return obj.get('message', {}).get('content', '')


def llm_respond(query, active_notes, nodes):
    """Send query + activated note context to LLM, get grounded response."""
    import urllib.request

    data, headers = _build_llm_payload(query, active_notes, nodes, stream=False)
    provider = _config.CONFIG.get('llm_provider', 'ollama')

    def _llm_call():
        req = urllib.request.Request(_config.OLLAMA_LLM_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=_config.CONFIG.get('llm_timeout', 300)) as resp:
            result = json.loads(resp.read())
            return _parse_llm_response(result, provider)

    try:
        raw = retry_with_backoff(_llm_call)
        # Sanitize: strip <pad> tokens, whitespace-only responses, and guardrail artefacts
        raw = re.sub(r'<pad>', '', raw).strip()
        # Filter known guardrail/refusal artefacts from free models
        guardrail_patterns = [
            r'^User Safety:\s*\w+$',
            r'^I cannot (help|assist) with',
            r'^As an AI',
        ]
        for pattern in guardrail_patterns:
            if re.match(pattern, raw, re.IGNORECASE):
                raw = ''
                break
        return raw if raw else '[no response from LLM]'
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_stream(query, active_notes, nodes):
    """Stream LLM response token-by-token.

    Supports Ollama native NDJSON and OpenAI-compatible SSE (Ollama Cloud
    or OpenRouter).
    Yields token strings as they arrive from the LLM.

    Phase 3.2: Online plasticity — the caller can use the streamed tokens
    to update Hebbian state progressively as the LLM generates.
    """
    import urllib.request

    data, headers = _build_llm_payload(query, active_notes, nodes, stream=True)
    provider = _config.CONFIG.get('llm_provider', 'ollama')

    req = urllib.request.Request(_config.OLLAMA_LLM_URL, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=_config.CONFIG.get('llm_timeout', 300)) as resp:
            buffer = b''
            for chunk in iter(lambda: resp.read(1), b''):
                buffer += chunk
                if buffer.endswith(b'\n'):
                    line = buffer.strip()
                    buffer = b''
                    if not line:
                        continue

                    if uses_openai_compatible_api(provider):
                        # OpenAI-compatible SSE: lines start with "data: "
                        if line.startswith(b'data: '):
                            line = line[6:]
                        if line == b'[DONE]':
                            break
                        try:
                            obj = json.loads(line)
                            token = _parse_llm_stream_token(obj, provider)
                            if token and token != '<pad>':
                                yield token
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Ollama NDJSON: one JSON object per line
                        try:
                            obj = json.loads(line)
                            if obj.get('done', False):
                                break
                            token = obj.get('message', {}).get('content', '')
                            if token and token != '<pad>':
                                yield token
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"[LLM stream error: {e}]"