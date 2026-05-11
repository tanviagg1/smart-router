"""
Tests for ModelRegistry.

Run: pytest tests/test_registry.py -v
"""

import pytest
from router.registry import ModelRegistry, ModelProfile


@pytest.fixture
def registry():
    return ModelRegistry()


class TestModelRegistry:

    def test_code_routes_to_codellama(self, registry):
        assert registry.get_model_for("code") == "codellama:7b"

    def test_simple_qa_routes_to_fast_model(self, registry):
        assert registry.get_model_for("simple_qa") == "llama3.2:3b"

    def test_complex_routes_to_capable_model(self, registry):
        assert registry.get_model_for("complex") == "llama3.1:8b"

    def test_default_routes_to_fallback(self, registry):
        result = registry.get_model_for("default")
        assert result in ["llama3.1:8b", "llama3.2:3b"]

    def test_unknown_task_returns_fallback(self, registry):
        result = registry.get_model_for("nonexistent_task_type")
        assert result == "llama3.1:8b"

    def test_get_profile_returns_correct_profile(self, registry):
        profile = registry.get_profile("codellama:7b")
        assert profile is not None
        assert profile.name == "codellama:7b"
        assert "code" in profile.task_types

    def test_get_profile_returns_none_for_unknown(self, registry):
        assert registry.get_profile("unknown:model") is None

    def test_list_profiles_returns_all(self, registry):
        profiles = registry.list_profiles()
        assert len(profiles) >= 3
        names = [p.name for p in profiles]
        assert "llama3.1:8b" in names
        assert "llama3.2:3b" in names
        assert "codellama:7b" in names

    def test_register_custom_model(self, registry):
        custom = ModelProfile(
            name="custom:model",
            speed="fast",
            task_types=["custom_task"],
            description="Custom test model",
        )
        registry.register(custom)
        assert registry.get_model_for("custom_task") == "custom:model"

    def test_profiles_have_required_fields(self, registry):
        for profile in registry.list_profiles():
            assert profile.name
            assert profile.speed in ("fast", "medium", "slow")
            assert isinstance(profile.task_types, list)
            assert profile.description


class TestRouterEngine:
    """Tests for RouterEngine that don't require Ollama."""

    def test_engine_classifies_and_selects_model(self):
        from unittest.mock import MagicMock, patch
        from router.engine import RouterEngine

        mock_response = "Binary search divides the array in half each time."

        with patch("ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = MagicMock(
                message=MagicMock(content=mock_response)
            )
            engine = RouterEngine()
            result = engine.route("Write a binary search in Python")

        assert result.task_type == "code"
        assert result.model_used == "codellama:7b"
        assert result.response == mock_response
        assert result.confidence > 0

    def test_engine_model_override_skips_classification(self):
        from unittest.mock import MagicMock, patch
        from router.engine import RouterEngine

        with patch("ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = MagicMock(
                message=MagicMock(content="response")
            )
            engine = RouterEngine(model_override="llama3.1:8b")
            result = engine.route("What is Python?")

        assert result.model_used == "llama3.1:8b"
        assert "override" in result.reason.lower()

    def test_engine_raises_on_empty_prompt(self):
        from router.engine import RouterEngine
        engine = RouterEngine()
        with pytest.raises(ValueError, match="empty"):
            engine.route("")

    def test_routing_result_str_contains_model(self):
        from unittest.mock import MagicMock, patch
        from router.engine import RouterEngine

        with patch("ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = MagicMock(
                message=MagicMock(content="answer")
            )
            engine = RouterEngine()
            result = engine.route("What is Python?")

        assert result.model_used in str(result)
