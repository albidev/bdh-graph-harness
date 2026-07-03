"""OpenRouter-specific payload builder and response parsers (OpenAI-compatible)."""

import json


def build_openrouter_payload(messages, stream, config):
    """Build OpenRouter-format request payload and headers.

    Returns (payload_bytes, headers).
    """
    payload = {
        "model": config['llm_model'],
        "messages": messages,
        "stream": stream,
        "temperature": config['llm_temperature'],
        "max_tokens": config['llm_max_ctx'],
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {config.get('openrouter_key', '')}",
        'HTTP-Referer': 'https://github.com/bdh-graph-harness',
        'X-Title': 'BDH Graph Harness',
    }
    return json.dumps(payload).encode(), headers


def parse_openrouter_response(result):
    """Parse a non-streaming OpenRouter response — returns content string."""
    choices = result.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('content', '[no response]')
    return '[no response]'


def parse_openrouter_stream_token(obj):
    """Parse a single OpenRouter SSE streaming chunk.

    Returns content string or None if no token.
    """
    choices = obj.get('choices', [])
    if choices:
        delta = choices[0].get('delta', {})
        content = delta.get('content', '')
        return content if content else None
    return None