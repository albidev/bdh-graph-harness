"""Ollama-specific payload builder and response parser."""

import json


def build_ollama_payload(messages, stream, config):
    """Build Ollama-format request payload and headers.

    Returns (payload_bytes, headers).
    """
    payload = {
        "model": config['llm_model'],
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": config['llm_temperature'],
            "num_ctx": config['llm_max_ctx'],
        },
    }
    headers = {'Content-Type': 'application/json'}
    return json.dumps(payload).encode(), headers


def parse_ollama_response(result):
    """Parse a non-streaming Ollama response — returns content string."""
    return result.get('message', {}).get('content', '[no response]')