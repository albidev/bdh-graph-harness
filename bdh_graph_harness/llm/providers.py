"""LLM provider dispatch — builds payloads, parses responses, and drives LLM calls.

Dispatcher functions (_build_llm_payload, _parse_llm_response, _parse_llm_stream_token)
delegate to ollama.py or openrouter.py based on CONFIG['llm_provider'].
"""

import json
import re

from bdh_graph_harness.config import CONFIG, OLLAMA_LLM_URL, retry_with_backoff
from bdh_graph_harness.llm.prompt import build_messages, format_context
from bdh_graph_harness.llm.ollama import build_ollama_payload, parse_ollama_response
from bdh_graph_harness.llm.openrouter import (
    build_openrouter_payload,
    parse_openrouter_response,
    parse_openrouter_stream_token,
)


def _build_llm_payload(query, active_notes, nodes, stream=False):
    """Build request payload + headers for the configured LLM provider.

    Returns (data_bytes, headers_dict).
    """
    messages = build_messages(query, active_notes, nodes)
    provider = CONFIG.get('llm_provider', 'ollama')

    if provider == 'openrouter':
        return build_openrouter_payload(messages, stream, CONFIG)
    else:
        return build_ollama_payload(messages, stream, CONFIG)


def _parse_llm_response(result, provider='ollama'):
    """Parse LLM response from either provider format."""
    if provider == 'openrouter':
        return parse_openrouter_response(result)
    else:
        return parse_ollama_response(result)


def _parse_llm_stream_token(obj, provider='ollama'):
    """Parse a single streaming chunk from either provider."""
    if provider == 'openrouter':
        return parse_openrouter_stream_token(obj)
    else:
        # Ollama: message.content
        if obj.get('done', False):
            return None
        return obj.get('message', {}).get('content', '')


def llm_respond(query, active_notes, nodes):
    """Send query + activated note context to LLM, get grounded response."""
    import urllib.request

    data, headers = _build_llm_payload(query, active_notes, nodes, stream=False)
    provider = CONFIG.get('llm_provider', 'ollama')

    def _llm_call():
        req = urllib.request.Request(OLLAMA_LLM_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=CONFIG.get('llm_timeout', 300)) as resp:
            result = json.loads(resp.read())
            return _parse_llm_response(result, provider)

    try:
        raw = retry_with_backoff(_llm_call)
        # Sanitize: strip <pad> tokens and whitespace-only responses
        raw = re.sub(r'<pad>', '', raw).strip()
        return raw if raw else '[no response from LLM]'
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_stream(query, active_notes, nodes):
    """Stream LLM response token-by-token.

    Supports both Ollama (NDJSON stream) and OpenRouter (SSE format).
    Yields token strings as they arrive from the LLM.

    Phase 3.2: Online plasticity — the caller can use the streamed tokens
    to update Hebbian state progressively as the LLM generates.
    """
    import urllib.request

    data, headers = _build_llm_payload(query, active_notes, nodes, stream=True)
    provider = CONFIG.get('llm_provider', 'ollama')

    req = urllib.request.Request(OLLAMA_LLM_URL, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=CONFIG.get('llm_timeout', 300)) as resp:
            buffer = b''
            for chunk in iter(lambda: resp.read(1), b''):
                buffer += chunk
                if buffer.endswith(b'\n'):
                    line = buffer.strip()
                    buffer = b''
                    if not line:
                        continue

                    if provider == 'openrouter':
                        # OpenRouter SSE: lines start with "data: "
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