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


def test_per_vault_llm_config_overrides_global_without_mutating_it(monkeypatch):
    """A local vault can override a cloud global config in isolation."""
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'deepseek-v4-flash:cloud',
        'llm_base_url': 'https://ollama.com/v1',
        'llm_api_key': 'cloud-key',
        'llm_temperature': 0.1,
        'llm_max_ctx': 4096,
    })
    from bdh_graph_harness.config import resolve_llm_config

    global_cfg = resolve_llm_config()
    local_cfg = resolve_llm_config({
        'llm': {
            'provider': 'ollama',
            'model': 'gemma4:26b-mlx',
            'base_url': 'http://127.0.0.1:11434',
        },
    })

    assert global_cfg['llm_provider'] == 'ollama-cloud'
    assert global_cfg['llm_endpoint'] == 'https://ollama.com/v1/chat/completions'
    assert local_cfg['llm_provider'] == 'ollama'
    assert local_cfg['llm_model'] == 'gemma4:26b-mlx'
    assert local_cfg['llm_endpoint'] == 'http://127.0.0.1:11434/api/chat'
    assert local_cfg['llm_api_key'] == ''
    assert bdh_config.CONFIG['llm_provider'] == 'ollama-cloud'


def test_build_payload_accepts_explicit_per_vault_config(mock_active_notes, mock_nodes):
    """Explicit vault config selects the local model even with cloud globals."""
    data, headers = harness._build_llm_payload(
        'test query',
        mock_active_notes,
        mock_nodes,
        config={
            'llm': {
                'provider': 'ollama',
                'model': 'gemma4:26b-mlx',
                'base_url': 'http://127.0.0.1:11434',
            },
        },
    )
    payload = json.loads(data)
    assert payload['model'] == 'gemma4:26b-mlx'
    assert 'options' in payload
    assert 'Authorization' not in headers


def test_omlx_provider_resolves_to_local_openai_compatible_endpoint():
    """oMLX is a local OpenAI-compatible provider with no cloud auth."""
    from bdh_graph_harness.config import resolve_llm_config

    config = resolve_llm_config({
        'llm_provider': 'omlx',
        'llm_model': 'qwen3.8-27b-oq4e-mtp',
    })

    assert config['llm_provider_label'] == 'oMLX'
    assert config['llm_transport'] == 'openai-compatible'
    assert config['llm_endpoint'] == 'http://127.0.0.1:8083/v1/chat/completions'
    assert config['llm_api_key'] == ''


def test_resolve_nous_portal_runtime_config(monkeypatch):
    """Nous Portal uses its OpenAI-compatible inference endpoint and bearer auth."""
    monkeypatch.setenv('NOUS_API_KEY', 'nous-test-key')
    config = bdh_config.resolve_llm_config({
        'llm_provider': 'nous',
        'llm_model': 'upstage/solar-pro4:free',
    })
    assert config['llm_provider_label'] == 'Nous Portal'
    assert config['llm_transport'] == 'openai-compatible'
    assert config['llm_base_url'] == 'https://inference-api.nousresearch.com/v1'
    assert config['llm_endpoint'] == 'https://inference-api.nousresearch.com/v1/chat/completions'
    assert config['llm_api_key'] == 'nous-test-key'


def test_resolve_nous_portal_from_hermes_auth_file(monkeypatch, tmp_path):
    """Standalone BDH can consume the short-lived Nous agent key without copying it."""
    auth_path = tmp_path / 'auth.json'
    auth_path.write_text(json.dumps({
        'providers': {
            'nous': {
                'agent_key': 'agent-key-from-hermes',
                'inference_base_url': 'https://inference-api.nousresearch.com/v1',
            },
        },
    }))
    monkeypatch.delenv('NOUS_API_KEY', raising=False)
    monkeypatch.setenv('NOUS_AUTH_FILE', str(auth_path))
    config = bdh_config.resolve_llm_config({
        'llm_provider': 'nous',
        'llm_model': 'upstage/solar-pro4:free',
    })
    assert config['llm_api_key'] == 'agent-key-from-hermes'

def test_build_payload_nous_omits_undeclared_response_format(mock_active_notes, mock_nodes, monkeypatch):
    """Nous's documented API is OpenAI-compatible but does not declare JSON mode."""
    monkeypatch.setenv('NOUS_API_KEY', 'nous-test-key')
    data, headers = harness._build_llm_payload(
        'test query', mock_active_notes, mock_nodes, stream=False,
        config={
            'llm_provider': 'nous',
            'llm_model': 'upstage/solar-pro4:free',
            'llm_temperature': 0.1,
            'llm_max_ctx': 4096,
        },
    )
    payload = json.loads(data)
    assert payload['model'] == 'upstage/solar-pro4:free'
    assert 'response_format' not in payload
    assert headers['Authorization'] == 'Bearer nous-test-key'
    assert headers['User-Agent'].startswith('BDH-Graph-Harness/')


def test_llm_respond_fails_over_cloud_nous_openrouter_then_omlx(mock_active_notes, mock_nodes, monkeypatch):
    """BDH walks the full cloud/provider/local chain in order."""
    import urllib.error
    import urllib.request

    monkeypatch.setenv('NOUS_API_KEY', 'nous-key')
    config = {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'deepseek-v4-flash:0731',
        'llm_base_url': 'https://ollama.com/v1',
        'llm_api_key': 'cloud-key',
        'llm_temperature': 0.1,
        'llm_max_ctx': 4096,
        'llm_fallbacks': [
            {'provider': 'nous', 'model': 'upstage/solar-pro4:free'},
            {'provider': 'openrouter', 'model': 'nvidia/nemotron-3-ultra-550b-a55b:free', 'api_key_env': 'OPENROUTER_API_KEY'},
            {'provider': 'omlx', 'model': 'qwen3.8-27b-oq4e-mtp', 'base_url': 'http://127.0.0.1:8083/v1'},
        ],
    }
    monkeypatch.setenv('OPENROUTER_API_KEY', 'openrouter-key')
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({'choices': [{'message': {'content': 'local response'}}]}).encode()

    def urlopen(req, timeout):
        calls.append((req.full_url, req.headers.get('Authorization')))
        if any(req.full_url.startswith(prefix) for prefix in ('https://ollama.com/', 'https://inference-api.nousresearch.com/', 'https://openrouter.ai/')):
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many Requests', {}, None)
        return Response()

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(bdh_providers, 'retry_with_backoff', lambda fn: fn())

    result = bdh_providers.llm_respond('test query', mock_active_notes, mock_nodes, config=config)
    assert result == 'local response'
    assert calls == [
        ('https://ollama.com/v1/chat/completions', 'Bearer cloud-key'),
        ('https://inference-api.nousresearch.com/v1/chat/completions', 'Bearer nous-key'),
        ('https://openrouter.ai/api/v1/chat/completions', 'Bearer openrouter-key'),
        ('http://127.0.0.1:8083/v1/chat/completions', None),
    ]


def test_llm_respond_fails_over_to_omlx_after_cloud_429(mock_active_notes, mock_nodes, monkeypatch):
    """A failed cloud completion is retried through the configured local oMLX fallback."""
    import urllib.error
    import urllib.request

    config = {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'deepseek-v4-flash:0731',
        'llm_base_url': 'https://ollama.com/v1',
        'llm_api_key': 'cloud-key',
        'llm_temperature': 0.1,
        'llm_max_ctx': 4096,
        'llm_fallbacks': [{
            'provider': 'omlx',
            'model': 'qwen3.8-27b-oq4e-mtp',
            'base_url': 'http://127.0.0.1:8083/v1',
        }],
    }
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({'choices': [{'message': {'content': 'local response'}}]}).encode()

    def urlopen(req, timeout):
        calls.append((req.full_url, req.headers.get('Authorization')))
        if req.full_url.startswith('https://ollama.com/'):
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many Requests', {}, None)
        return Response()

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(bdh_providers, 'retry_with_backoff', lambda fn: fn())

    result = bdh_providers.llm_respond('test query', mock_active_notes, mock_nodes, config=config)

    assert result == 'local response'
    assert calls == [
        ('https://ollama.com/v1/chat/completions', 'Bearer cloud-key'),
        ('http://127.0.0.1:8083/v1/chat/completions', None),
    ]


def test_llm_stream_fails_over_to_omlx_before_emitting_tokens(mock_active_notes, mock_nodes, monkeypatch):
    """Streaming switches candidates only when the primary fails before output."""
    import urllib.error
    import urllib.request

    config = {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'deepseek-v4-flash:0731',
        'llm_base_url': 'https://ollama.com/v1',
        'llm_api_key': 'cloud-key',
        'llm_temperature': 0.1,
        'llm_max_ctx': 4096,
        'llm_fallbacks': [{
            'provider': 'omlx',
            'model': 'qwen3.8-27b-oq4e-mtp',
            'base_url': 'http://127.0.0.1:8083/v1',
        }],
    }
    calls = []

    class Response:
        def __init__(self):
            self.data = (
                b'data: ' + json.dumps({'choices': [{'delta': {'content': 'local'}}]}).encode()
                + b'\n\n'
                + b'data: ' + json.dumps({'choices': [{'delta': {'content': ' stream'}}]}).encode()
                + b'\n\n'
                + b'data: [DONE]\n\n'
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            if not self.data:
                return b''
            chunk, self.data = self.data[:size], self.data[size:]
            return chunk

    def urlopen(req, timeout):
        calls.append((req.full_url, req.headers.get('Authorization')))
        if req.full_url.startswith('https://ollama.com/'):
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many Requests', {}, None)
        return Response()

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)

    result = list(bdh_providers.llm_stream('test query', mock_active_notes, mock_nodes, config=config))

    assert ''.join(result) == 'local stream'
    assert calls == [
        ('https://ollama.com/v1/chat/completions', 'Bearer cloud-key'),
        ('http://127.0.0.1:8083/v1/chat/completions', None),
    ]


def test_local_only_gate_rejects_cloud_provider(monkeypatch):
    monkeypatch.setattr(bdh_config, 'CONFIG', {
        'llm_provider': 'ollama-cloud',
        'llm_model': 'cloud-model',
        'llm_base_url': 'https://ollama.com/v1',
    })
    from bdh_graph_harness.config import resolve_llm_config

    with pytest.raises(ValueError, match='local_only=true'):
        resolve_llm_config({'llm': {'local_only': True}})


def test_local_only_gate_rejects_non_local_ollama_endpoint():
    from bdh_graph_harness.config import resolve_llm_config

    with pytest.raises(ValueError, match='localhost'):
        resolve_llm_config({
            'llm': {
                'local_only': True,
                'provider': 'ollama',
                'base_url': 'https://remote.example/ollama',
            },
        })


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


# ---------------------------------------------------------------------------
# Level B: graph-aware prompt
# ---------------------------------------------------------------------------

def _build_messages(**overrides):
    from bdh_graph_harness.llm.prompt import build_messages
    base = {
        "active_notes": {'wiki/apple': 0.8, 'wiki/banana': 0.6},
        "nodes": {
            'wiki/apple': {'id': 'wiki/apple', 'title': 'Apple', 'text': 'Apple is a fruit.'},
            'wiki/banana': {'id': 'wiki/banana', 'title': 'Banana', 'text': 'Banana is yellow.'},
        },
    }
    base.update(overrides)
    return build_messages("what is apple?", **base)


def test_levelb_defaults_to_legacy_when_no_extra_signals():
    """Without state/associative_context, the prompt stays the legacy RAG prompt."""
    msgs = _build_messages()
    sys_prompt = msgs[0]["content"]
    user_prompt = msgs[1]["content"]
    # Legacy prompt does NOT contain graph-reasoning guidance.
    assert "Associative context" not in user_prompt
    assert "quality/dormancy" not in sys_prompt
    assert "learned to associate" not in sys_prompt
    # It still carries the activated notes.
    assert "### Apple" in user_prompt


def test_levelb_injects_associative_context_block():
    """associative_context appears as a labelled, inferred lane in the prompt."""
    assoc = [
        {"id": "wiki/banana", "relationship": "co-activated", "weight": 0.9},
        {"id": "wiki/kiwi", "weight": 0.5},
    ]
    msgs = _build_messages(associative_context=assoc)
    user_prompt = msgs[1]["content"]
    sys_prompt = msgs[0]["content"]
    assert "## Associative context" in user_prompt
    assert "Banana" in user_prompt
    # Inferred-lane guidance present in system prompt.
    assert "learned to associate" in sys_prompt


def test_levelb_annotates_node_quality_when_state_given():
    """dormant/quality tags are added when a state dict is supplied."""
    state = {
        "dormant_nodes": {"wiki/banana"},
        "node_quality": {"wiki/apple": {"score": 0.9}},
    }
    msgs = _build_messages(state=state)
    user_prompt = msgs[1]["content"]
    assert "dormant" in user_prompt
    assert "quality=0.9" in user_prompt

def test_source_llm_override_isolated_from_global_config(monkeypatch):
    """A source override changes only the selected runtime configuration."""
    from bdh_graph_harness.config import resolve_llm_config_for_source

    base = {
        "llm_provider": "ollama-cloud",
        "llm_model": "deepseek-v4-pro",
        "llm_base_url": "https://ollama.com/v1",
        "llm_temperature": 0.3,
        "llm_max_ctx": 4096,
        "llm_source_overrides": {
            "session_synthesis": {
                "provider": "ollama-cloud",
                "model": "deepseek-v4-flash:cloud",
                "temperature": 0.1,
                "reasoning_effort": "low",
                "thinking": "enabled",
            },
        },
    }

    synthesis = resolve_llm_config_for_source(base, "session_synthesis")
    normal = resolve_llm_config_for_source(base, "assistant_response")

    assert synthesis["llm_model"] == "deepseek-v4-flash:cloud"
    assert synthesis["llm_temperature"] == 0.1
    assert synthesis["llm_reasoning_effort"] == "low"
    assert synthesis["llm_thinking"] == "enabled"
    assert normal["llm_model"] == "deepseek-v4-pro"
    assert base["llm_model"] == "deepseek-v4-pro"


def test_openai_payload_carries_low_reasoning_for_source_override(
    mock_active_notes, mock_nodes,
):
    """DeepSeek source overrides produce the explicit low-effort wire fields."""
    config = {
        "llm_provider": "ollama-cloud",
        "llm_model": "deepseek-v4-flash:cloud",
        "llm_temperature": 0.1,
        "llm_max_ctx": 4096,
        "llm_reasoning_effort": "low",
        "llm_thinking": "enabled",
    }
    data, _ = bdh_providers._build_llm_payload(
        "synthesis", mock_active_notes, mock_nodes, config=config,
    )
    payload = json.loads(data)
    assert payload["model"] == "deepseek-v4-flash:cloud"
    assert payload["temperature"] == 0.1
    assert payload["reasoning_effort"] == "low"
    assert payload["thinking"] == {"type": "enabled"}
