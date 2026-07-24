"""OpenAI-compatible payload and response helpers.

Used by providers that expose the Chat Completions contract, including
OpenRouter and Ollama Cloud. The transport is shared; the provider identity is
kept in configuration and diagnostics instead of being inferred from the
wire format.
"""

import json


def build_openai_compatible_payload(messages, stream, config):
    """Build an OpenAI-compatible chat completion request."""
    payload = {
        "model": config['llm_model'],
        "messages": messages,
        "stream": stream,
        "temperature": config['llm_temperature'],
        "max_tokens": config.get('llm_max_tokens', min(config['llm_max_ctx'], 2048)),
    }
    api_key = config.get('llm_api_key') or config.get('openrouter_key', '')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}",
        'User-Agent': 'BDH-Graph-Harness/1.0',
    }
    if config.get('llm_provider') == 'openrouter':
        headers.update({
            'HTTP-Referer': 'https://github.com/bdh-graph-harness',
            'X-Title': 'BDH Graph Harness',
        })
    return json.dumps(payload).encode(), headers


def parse_openai_compatible_response(result):
    """Parse a non-streaming Chat Completions response."""
    choices = result.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('content', '[no response]')
    return '[no response]'


def parse_openai_compatible_stream_token(obj):
    """Parse one Chat Completions SSE chunk."""
    choices = obj.get('choices', [])
    if choices:
        delta = choices[0].get('delta', {})
        content = delta.get('content', '')
        return content if content else None
    return None
