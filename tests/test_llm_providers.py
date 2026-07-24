"""Tests for LLM provider support (Phase 3 — OpenRouter integration).

Tests the _build_llm_payload, _parse_llm_response, and _parse_llm_stream_token
helper functions that abstract provider differences (Ollama vs OpenRouter).
"""
import json
import pytest
import harness
from bdh_graph_harness import config as bdh_config
from bdh_graph_harness.llm import providers as bdh_providers


@pytest.fixture
def mock_active_notes():
    return {'wiki/apple': 0.8, 'wiki/banana': 0.6}


@pytest.fixture
def mock_nodes():
    return {
        'wiki/apple': {'id': 'wiki/apple', 'title': 'Apple', 'text': 'Apple is a fruit.'},
        'wiki/banana': {'id': 'wiki/banana', 'title': 'Banana', 'text': 'Banana is yellow.'},
    }


# ---------------------------------------------------------------------------
# _build_llm_payload
# ---------------------------------------------------------------------------

def test_build_payload_ollama_format(mock_active_notes, mock_nodes, monkeypatch):
    """Ollama payload uses 'options' key for params."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama',
        'llm_model': 'gemma4:12b-mlx',
        'llm_temperature': 0.3,
        'llm_max_ctx': 4096,
    })
    data, headers = harness._build_llm_payload('test query', mock_active_notes, mock_nodes, stream=False)
    payload = json.loads(data)
    assert payload['model'] == 'gemma4:12b-mlx'
    assert payload['stream'] is False
    assert 'options' in payload
    assert payload['options']['temperature'] == 0.3
    assert payload['options']['num_ctx'] == 4096
    assert 'Authorization' not in headers
    assert headers['Content-Type'] == 'application/json'


def test_build_payload_openrouter_format(mock_active_notes, mock_nodes, monkeypatch):
    """OpenRouter payload uses 'temperature' and 'max_tokens' at top level."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'openrouter',
        'llm_model': 'openrouter/free',
        'llm_temperature': 0.3,
        'llm_max_ctx': 4096,
        'openrouter_key': 'sk-test-key-123',
    })
    data, headers = harness._build_llm_payload('test query', mock_active_notes, mock_nodes, stream=True)
    payload = json.loads(data)
    assert payload['model'] == 'openrouter/free'
    assert payload['stream'] is True
    assert payload['temperature'] == 0.3
    assert payload['max_tokens'] == 2048  # defaults to min(llm_max_ctx, 2048)
    assert 'options' not in payload
    assert headers['Authorization'] == 'Bearer sk-test-key-123'
    assert headers['HTTP-Referer'] == 'https://github.com/bdh-graph-harness'
    assert headers['X-Title'] == 'BDH Graph Harness'


def test_build_payload_ollama_cloud_uses_canonical_openai_compatible_config(
    mock_active_notes, mock_nodes, monkeypatch,
):
    """Ollama Cloud uses its own provider name, not the OpenRouter code path label."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'deepseek-v4-flash:cloud',
        'llm_temperature': 0.1,
        'llm_max_ctx': 4096,
        'llm_api_key': 'ollama-cloud-test-key',
    })
    data, headers = harness._build_llm_payload(
        'test query', mock_active_notes, mock_nodes, stream=True,
    )
    payload = json.loads(data)
    assert payload['model'] == 'deepseek-v4-flash:cloud'
    assert payload['temperature'] == 0.1
    assert headers['Authorization'] == 'Bearer ollama-cloud-test-key'
    assert 'HTTP-Referer' not in headers


def test_config_reports_ollama_cloud_runtime_without_openrouter_alias(monkeypatch):
    """Canonical config exposes the actual provider and endpoint semantics."""
    import tempfile, os

    original_config = bdh_config.CONFIG.copy()
    original_llm_url = bdh_config.OLLAMA_LLM_URL
    monkeypatch.setenv('TEST_OLLAMA_CLOUD_KEY', 'ollama-cloud-secret')
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(
                'llm_provider: ollama-cloud\n'
                'llm_base_url: https://ollama.com/v1\n'
                'llm_api_key: ${TEST_OLLAMA_CLOUD_KEY}\n'
                'llm_model: deepseek-v4-flash:cloud\n'
            )
            f.flush()
            config = harness.load_config(f.name)
            os.unlink(f.name)

        assert config['llm_provider'] == 'ollama-cloud'
        assert config['llm_transport'] == 'openai-compatible'
        assert config['llm_provider_label'] == 'Ollama Cloud'
        assert bdh_config.OLLAMA_LLM_URL == 'https://ollama.com/v1/chat/completions'
    finally:
        bdh_config.CONFIG.clear()
        bdh_config.CONFIG.update(original_config)
        bdh_config.OLLAMA_LLM_URL = original_llm_url


def test_build_payload_messages_always_present(mock_active_notes, mock_nodes, monkeypatch):
    """Both providers get messages array with system + user roles."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama',
        'llm_model': 'test',
        'llm_temperature': 0.3,
        'llm_max_ctx': 4096,
    })
    data, _ = harness._build_llm_payload('hello', mock_active_notes, mock_nodes)
    payload = json.loads(data)
    assert len(payload['messages']) == 2
    assert payload['messages'][0]['role'] == 'system'
    assert payload['messages'][1]['role'] == 'user'
    assert 'hello' in payload['messages'][1]['content']


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

def test_parse_response_ollama():
    """Ollama response: message.content."""
    result = {'message': {'content': 'Hello from Ollama'}}
    assert harness._parse_llm_response(result, 'ollama') == 'Hello from Ollama'


def test_parse_response_openrouter():
    """OpenRouter response: choices[0].message.content."""
    result = {'choices': [{'message': {'content': 'Hello from OpenRouter'}}]}
    assert harness._parse_llm_response(result, 'openrouter') == 'Hello from OpenRouter'


def test_parse_response_openrouter_empty():
    """OpenRouter with no choices returns default."""
    result = {'choices': []}
    assert harness._parse_llm_response(result, 'openrouter') == '[no response]'


def test_parse_response_ollama_empty():
    """Ollama with no message returns default."""
    result = {}
    assert harness._parse_llm_response(result, 'ollama') == '[no response]'


# ---------------------------------------------------------------------------
# _parse_llm_stream_token
# ---------------------------------------------------------------------------

def test_parse_stream_token_ollama():
    """Ollama streaming: message.content."""
    obj = {'message': {'content': 'Hello'}, 'done': False}
    assert harness._parse_llm_stream_token(obj, 'ollama') == 'Hello'


def test_parse_stream_token_ollama_done():
    """Ollama done signal returns None."""
    obj = {'done': True}
    assert harness._parse_llm_stream_token(obj, 'ollama') is None


def test_parse_stream_token_openrouter():
    """OpenRouter streaming: choices[0].delta.content."""
    obj = {'choices': [{'delta': {'content': 'World'}}]}
    assert harness._parse_llm_stream_token(obj, 'openrouter') == 'World'


def test_parse_stream_token_openrouter_empty_delta():
    """OpenRouter with empty delta returns None."""
    obj = {'choices': [{'delta': {}}]}
    assert harness._parse_llm_stream_token(obj, 'openrouter') is None


def test_parse_stream_token_openrouter_no_choices():
    """OpenRouter with no choices returns None."""
    obj = {'choices': []}
    assert harness._parse_llm_stream_token(obj, 'openrouter') is None


# ---------------------------------------------------------------------------
# llm_respond sanitization
# ---------------------------------------------------------------------------

def test_llm_respond_strips_pad_tokens(mock_active_notes, mock_nodes, monkeypatch):
    """llm_respond strips <pad> tokens from response."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama',
        'llm_model': 'test',
        'llm_temperature': 0.3,
        'llm_max_ctx': 4096,
        'llm_timeout': 10,
    })
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'http://fake')
    monkeypatch.setattr(bdh_providers, 'retry_with_backoff', lambda fn: '<pad><pad>Hello world<pad>')
    result = harness.llm_respond('test', mock_active_notes, mock_nodes)
    assert '<pad>' not in result
    assert result == 'Hello world'


def test_llm_respond_empty_after_strip(mock_active_notes, mock_nodes, monkeypatch):
    """llm_respond returns fallback when only <pad> tokens."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama',
        'llm_model': 'test',
        'llm_temperature': 0.3,
        'llm_max_ctx': 4096,
        'llm_timeout': 10,
    })
    monkeypatch.setattr(bdh_config, 'OLLAMA_LLM_URL', 'http://fake')
    monkeypatch.setattr(bdh_providers, 'retry_with_backoff', lambda fn: '<pad><pad><pad>')
    result = harness.llm_respond('test', mock_active_notes, mock_nodes)
    assert result == '[no response from LLM]'


# ---------------------------------------------------------------------------
# Config env var expansion
# ---------------------------------------------------------------------------

def test_config_env_var_expansion(monkeypatch):
    """load_config expands ${ENV_VAR} syntax."""
    import tempfile, os
    monkeypatch.setenv('TEST_BDH_KEY', 'secret-key-456')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("openrouter_key: ${TEST_BDH_KEY}\n")
        f.flush()
        config = harness.load_config(f.name)
        os.unlink(f.name)
    assert config['openrouter_key'] == 'secret-key-456'