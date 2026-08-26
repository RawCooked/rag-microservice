from rag_core.config import Settings


def test_defaults_are_safe():
    settings = Settings(_env_file=None)
    assert settings.anthropic_model == "claude-opus-5"
    assert settings.qdrant_url == ""  # embedded mode by default
    assert settings.top_k <= settings.fetch_k


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("RAG_INSTANCE_NAME", "api-2")
    monkeypatch.setenv("RAG_QDRANT_URL", "http://qdrant:6333")
    settings = Settings(_env_file=None)
    assert settings.instance_name == "api-2"
    assert settings.qdrant_url == "http://qdrant:6333"


def test_anthropic_key_uses_standard_env_name(monkeypatch):
    # The key must come from ANTHROPIC_API_KEY (the SDK's standard name),
    # not RAG_ANTHROPIC_API_KEY.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key == "sk-test-123"
