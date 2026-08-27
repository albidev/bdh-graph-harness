"""LLM provider dispatch — builds payloads, parses responses, and drives LLM calls.

Dispatcher functions (_build_llm_payload, _parse_llm_response, _parse_llm_stream_token)
delegate to ollama.py or openrouter.py based on CONFIG['llm_provider'].
"""

import json
import re
import logging

from bdh_graph_harness.config import (
    retry_with_backoff,
    resolve_llm_candidates,
    resolve_llm_config,
)
import bdh_graph_harness.config as _config
from bdh_graph_harness.llm.prompt import build_messages, format_context
from bdh_graph_harness.llm.ollama import build_ollama_payload, parse_ollama_response
from bdh_graph_harness.llm.openai_compatible import (
    build_openai_compatible_payload,
    parse_openai_compatible_response,
    parse_openai_compatible_stream_token,
)


logger = logging.getLogger('bdh.llm')


OPENAI_COMPATIBLE_PROVIDERS = frozenset({'openrouter', 'ollama-cloud', 'omlx'})


def uses_openai_compatible_api(provider=None, config=None):
    """Return whether *provider* speaks the Chat Completions contract."""
    if isinstance(provider, dict) and config is None:
        config = provider
        provider = None
    if config is not None:
        provider = config.get('llm_provider', 'ollama')
    provider = provider or _config.CONFIG.get('llm_provider', 'ollama')
    return provider in OPENAI_COMPATIBLE_PROVIDERS


def _build_llm_payload(query, active_notes, nodes, stream=False, config=None,
                       state=None, associative_context=None):
    """Build request payload + headers for the configured LLM provider.

    Returns (data_bytes, headers_dict).
    """
    runtime_config = resolve_llm_config(config)
    messages = build_messages(
        query, active_notes, nodes,
        state=state, associative_context=associative_context,
    )
    provider = runtime_config.get('llm_provider', 'ollama')

    if uses_openai_compatible_api(config=runtime_config):
        return build_openai_compatible_payload(messages, stream, runtime_config)
    else:
        return build_ollama_payload(messages, stream, runtime_config)


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


def _request_completion(messages, config=None, *, stream=False, json_mode=False):
    """Request one completion, falling through configured candidates on error.

    Returns ``(text, runtime_config)``. Retries are applied independently to
    each candidate; a fallback is attempted only after the candidate exhausts
    its retries. Secrets never appear in the failover log.
    """
    candidates = resolve_llm_candidates(config)
    if config is None and _config.OLLAMA_LLM_URL:
        # Preserve the legacy global URL override for the primary only.
        candidates[0]['llm_endpoint'] = _config.OLLAMA_LLM_URL

    last_error = None
    for index, runtime_config in enumerate(candidates):
        provider = runtime_config.get('llm_provider', 'ollama')
        messages_payload = messages
        if uses_openai_compatible_api(config=runtime_config):
            from bdh_graph_harness.llm.openai_compatible import build_openai_compatible_payload
            data, headers = build_openai_compatible_payload(
                messages_payload, stream, runtime_config, json_mode=json_mode,
            )
        else:
            data, headers = build_ollama_payload(
                messages_payload, stream, runtime_config, json_mode=json_mode,
            )

        def _call():
            import urllib.request
            req = urllib.request.Request(runtime_config['llm_endpoint'], data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=runtime_config.get('llm_timeout', 300)) as resp:
                result = json.loads(resp.read())
                return _parse_llm_response(result, provider)

        try:
            return retry_with_backoff(_call), runtime_config
        except Exception as exc:
            last_error = exc
            if index + 1 < len(candidates):
                next_config = candidates[index + 1]
                logger.warning(
                    'LLM candidate failed (%s/%s): provider=%s model=%s error=%s; '
                    'failing over to provider=%s model=%s',
                    index + 1, len(candidates), provider,
                    runtime_config.get('llm_model'), exc,
                    next_config.get('llm_provider'), next_config.get('llm_model'),
                )
    if last_error is not None:
        raise last_error
    raise RuntimeError('No valid LLM candidates configured')


def llm_respond(query, active_notes, nodes, config=None,
                state=None, associative_context=None):
    """Send query + activated note context to LLM, get grounded response."""
    import urllib.request

    try:
        messages = build_messages(
            query, active_notes, nodes,
            state=state, associative_context=associative_context,
        )
        raw, _runtime_config = _request_completion(messages, config, stream=False)
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
        logger.error(f"LLM respond failed: {e}", exc_info=True)
        return f"[LLM error: {e}]"


def llm_stream(query, active_notes, nodes, config=None,
               state=None, associative_context=None):
    """Stream LLM response token-by-token.

    Supports Ollama native NDJSON and OpenAI-compatible SSE (Ollama Cloud
    or OpenRouter).
    Yields token strings as they arrive from the LLM.

    Phase 3.2: Online plasticity — the caller can use the streamed tokens
    to update Hebbian state progressively as the LLM generates.
    """
    import urllib.request

    messages = build_messages(
        query, active_notes, nodes,
        state=state, associative_context=associative_context,
    )
    candidates = resolve_llm_candidates(config)
    if config is None and _config.OLLAMA_LLM_URL:
        candidates[0]['llm_endpoint'] = _config.OLLAMA_LLM_URL

    for index, runtime_config in enumerate(candidates):
        provider = runtime_config.get('llm_provider', 'ollama')
        if uses_openai_compatible_api(config=runtime_config):
            data, headers = build_openai_compatible_payload(messages, True, runtime_config)
        else:
            data, headers = build_ollama_payload(messages, True, runtime_config)
        req = urllib.request.Request(runtime_config['llm_endpoint'], data=data, headers=headers)
        yielded = False
        try:
            timeout = runtime_config.get('llm_timeout', 300)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                                    yielded = True
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
                                    yielded = True
                                    yield token
                            except json.JSONDecodeError:
                                continue
            return
        except Exception as exc:
            if index + 1 < len(candidates) and not yielded:
                next_config = candidates[index + 1]
                logger.warning(
                    'LLM stream candidate failed: provider=%s model=%s error=%s; '
                    'failing over to provider=%s model=%s',
                    provider, runtime_config.get('llm_model'), exc,
                    next_config.get('llm_provider'), next_config.get('llm_model'),
                )
                continue
            logger.error(f"LLM stream failed: {exc}", exc_info=True)
            yield f"[LLM stream error: {exc}]"
            return