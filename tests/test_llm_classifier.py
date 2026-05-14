"""
Tests for LLMClassifier and Phase 2 RouterEngine behavior.

Unit tests mock the OllamaClient so no real Ollama instance is needed.
Integration tests are marked @pytest.mark.integration and require
llama3.2:3b to be running locally.

Run unit tests only:
    pytest tests/test_llm_classifier.py -v -m "not integration"

Run all including integration:
    pytest tests/test_llm_classifier.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from router.llm_classifier import LLMClassifier, LLMClassifierError, CONFIDENCE_THRESHOLD
from router.classifier import ClassificationResult
from router.engine import RouterEngine, RoutingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_response(task_type: str, confidence: float, reason: str) -> str:
    """Build a valid JSON string that mimics a real LLM response."""
    return f'{{"task_type": "{task_type}", "confidence": {confidence}, "reason": "{reason}"}}'


# ---------------------------------------------------------------------------
# LLMClassifier — unit tests (mocked OllamaClient)
# ---------------------------------------------------------------------------

class TestLLMClassifierParsing:
    """Tests for _parse_response — no real LLM calls, tests JSON parsing logic."""

    def setup_method(self):
        self.clf = LLMClassifier()

    def test_parses_valid_code_response(self):
        raw = make_llm_response("code", 0.9, "Code generation task")
        result = self.clf._parse_response(raw, "write a sort")
        assert result.task_type == "code"
        assert result.confidence == 0.9

    def test_parses_simple_qa_response(self):
        raw = make_llm_response("simple_qa", 0.85, "Short factual question")
        result = self.clf._parse_response(raw, "What is recursion?")
        assert result.task_type == "simple_qa"

    def test_parses_complex_response(self):
        raw = make_llm_response("complex", 0.8, "Requires deep reasoning")
        result = self.clf._parse_response(raw, "Explain CAP theorem")
        assert result.task_type == "complex"

    def test_parses_default_response(self):
        raw = make_llm_response("default", 0.5, "No clear match")
        result = self.clf._parse_response(raw, "random prompt")
        assert result.task_type == "default"

    def test_reason_prefixed_with_llm_tag(self):
        # Routing results should show [LLM] so users know it came from LLM classifier
        raw = make_llm_response("code", 0.9, "Code task")
        result = self.clf._parse_response(raw, "write code")
        assert result.reason.startswith("[LLM]")

    def test_strips_markdown_code_fences(self):
        # Some models wrap JSON in ```json ... ``` — we should handle that
        raw = '```json\n{"task_type": "code", "confidence": 0.88, "reason": "code"}\n```'
        result = self.clf._parse_response(raw, "write a function")
        assert result.task_type == "code"

    def test_clamps_confidence_above_1(self):
        # Model might return confidence > 1.0 — clamp it
        raw = '{"task_type": "code", "confidence": 1.5, "reason": "very sure"}'
        result = self.clf._parse_response(raw, "write code")
        assert result.confidence == 1.0

    def test_clamps_confidence_below_0(self):
        raw = '{"task_type": "simple_qa", "confidence": -0.2, "reason": "unsure"}'
        result = self.clf._parse_response(raw, "what is x?")
        assert result.confidence == 0.0

    def test_raises_on_invalid_task_type(self):
        raw = '{"task_type": "unknown_type", "confidence": 0.9, "reason": "test"}'
        with pytest.raises(LLMClassifierError, match="unknown task_type"):
            self.clf._parse_response(raw, "some prompt")

    def test_raises_on_missing_json(self):
        with pytest.raises(LLMClassifierError, match="No JSON found"):
            self.clf._parse_response("I cannot classify this.", "some prompt")

    def test_raises_on_malformed_json(self):
        # Has braces (so regex finds it) but invalid JSON content inside
        with pytest.raises(LLMClassifierError, match="Failed to parse"):
            self.clf._parse_response('{"task_type": "code", "confidence": }', "prompt")


class TestLLMClassifierCallsMocked:
    """Tests for classify() with a mocked OllamaClient — no Ollama needed."""

    def test_classify_calls_ollama_with_prompt(self):
        clf = LLMClassifier()
        clf.client = MagicMock()
        clf.client.generate.return_value = make_llm_response("code", 0.9, "Code task")

        result = clf.classify("Write a sort function")

        # Verify the OllamaClient was called with the right model
        clf.client.generate.assert_called_once()
        call_kwargs = clf.client.generate.call_args
        assert call_kwargs.kwargs["model"] == "llama3.2:3b"
        assert call_kwargs.kwargs["temperature"] == 0.0  # must be deterministic

    def test_classify_returns_classification_result(self):
        clf = LLMClassifier()
        clf.client = MagicMock()
        clf.client.generate.return_value = make_llm_response("complex", 0.82, "Deep reasoning")

        result = clf.classify("Explain the CAP theorem in depth")

        assert isinstance(result, ClassificationResult)
        assert result.task_type == "complex"
        assert result.confidence == 0.82

    def test_classify_raises_on_ollama_failure(self):
        clf = LLMClassifier()
        clf.client = MagicMock()
        clf.client.generate.side_effect = Exception("Connection refused")

        with pytest.raises(LLMClassifierError, match="LLM call failed"):
            clf.classify("any prompt")


# ---------------------------------------------------------------------------
# RouterEngine — Phase 2 behavior (mocked)
# ---------------------------------------------------------------------------

class TestRouterEngineFallback:
    """Tests that RouterEngine falls back to rule-based when LLM fails."""

    def test_falls_back_to_rule_based_when_llm_fails(self):
        engine = RouterEngine(use_llm_classifier=True)

        # Make LLM classifier raise an error
        engine.llm_classifier = MagicMock()
        engine.llm_classifier.classify.side_effect = LLMClassifierError("model not found")

        # Mock OllamaClient so we don't need real Ollama
        engine.client = MagicMock()
        engine.client.generate.return_value = "Test response"

        result = engine.route("What is recursion?")

        # Should have used rule-based fallback
        assert result.classifier_used == "rule_based"
        assert result.response == "Test response"

    def test_uses_llm_classifier_when_available(self):
        engine = RouterEngine(use_llm_classifier=True)

        # Make LLM classifier succeed
        engine.llm_classifier = MagicMock()
        engine.llm_classifier.classify.return_value = ClassificationResult(
            task_type="code", confidence=0.9, reason="[LLM] Code task"
        )
        engine.client = MagicMock()
        engine.client.generate.return_value = "def binary_search(): ..."

        result = engine.route("Write a binary search")

        assert result.classifier_used == "llm"
        assert result.task_type == "code"

    def test_rule_based_only_when_use_llm_false(self):
        # use_llm_classifier=False disables LLM entirely — useful for testing
        engine = RouterEngine(use_llm_classifier=False)
        engine.client = MagicMock()
        engine.client.generate.return_value = "Paris"

        result = engine.route("What is the capital of France?")

        assert result.classifier_used == "rule_based"
        assert engine.llm_classifier is None


class TestRouterEngineEscalation:
    """Tests that low-confidence results get escalated to llama3.1:8b."""

    def test_escalates_when_confidence_below_threshold(self):
        engine = RouterEngine(use_llm_classifier=True)

        # Return a low-confidence classification
        engine.llm_classifier = MagicMock()
        engine.llm_classifier.classify.return_value = ClassificationResult(
            task_type="simple_qa",
            confidence=CONFIDENCE_THRESHOLD - 0.1,  # just below threshold
            reason="[LLM] unsure",
        )
        engine.client = MagicMock()
        engine.client.generate.return_value = "response"

        result = engine.route("some ambiguous prompt")

        assert result.escalated is True
        assert result.model_used == "llama3.1:8b"  # escalation model

    def test_no_escalation_when_confidence_above_threshold(self):
        engine = RouterEngine(use_llm_classifier=True)

        engine.llm_classifier = MagicMock()
        engine.llm_classifier.classify.return_value = ClassificationResult(
            task_type="code",
            confidence=CONFIDENCE_THRESHOLD + 0.1,  # above threshold
            reason="[LLM] Clear code task",
        )
        engine.client = MagicMock()
        engine.client.generate.return_value = "def foo(): ..."

        result = engine.route("Write a function")

        assert result.escalated is False
        assert result.model_used == "codellama:7b"  # correct model from registry

    def test_escalation_flag_in_result_str(self):
        engine = RouterEngine(use_llm_classifier=False)
        engine.client = MagicMock()
        engine.client.generate.return_value = "response"

        # Manually trigger escalation by setting a low-confidence rule result
        engine.rule_classifier = MagicMock()
        engine.rule_classifier.classify.return_value = ClassificationResult(
            task_type="default",
            confidence=0.3,  # very low
            reason="no signals",
        )

        result = engine.route("vague prompt")

        assert result.escalated is True
        assert "escalated" in str(result)


class TestRouterEngineModelOverride:
    """Tests that model_override bypasses classification entirely."""

    def test_override_skips_classification(self):
        engine = RouterEngine(model_override="llama3.1:8b")
        engine.llm_classifier = MagicMock()
        engine.client = MagicMock()
        engine.client.generate.return_value = "response"

        result = engine.route("any prompt")

        # LLM classifier should never be called
        engine.llm_classifier.classify.assert_not_called()
        assert result.model_used == "llama3.1:8b"
        assert result.classifier_used == "override"


# ---------------------------------------------------------------------------
# Integration tests — require llama3.2:3b running in Ollama
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLLMClassifierIntegration:
    """
    Real Ollama calls — skipped in CI, run manually with:
        pytest tests/test_llm_classifier.py -v -m integration
    """

    def setup_method(self):
        self.clf = LLMClassifier()

    def test_classifies_code_prompt(self):
        result = self.clf.classify("Write a binary search algorithm in Python")
        assert result.task_type == "code"
        assert result.confidence > CONFIDENCE_THRESHOLD

    def test_classifies_simple_qa_prompt(self):
        result = self.clf.classify("What is recursion?")
        assert result.task_type == "simple_qa"

    def test_classifies_complex_prompt(self):
        result = self.clf.classify("Explain the CAP theorem in depth with examples")
        assert result.task_type == "complex"

    def test_result_has_reason(self):
        result = self.clf.classify("What is Python?")
        assert result.reason.startswith("[LLM]")
        assert len(result.reason) > 5


@pytest.mark.integration
class TestRouterEngineIntegration:
    """Full end-to-end routing — requires Ollama with all models running."""

    def test_routes_code_prompt_to_codellama(self):
        engine = RouterEngine()
        result = engine.route("Write a bubble sort in Python")
        assert result.model_used == "codellama:7b"
        assert result.classifier_used == "llm"

    def test_routes_simple_qa_to_small_model(self):
        engine = RouterEngine()
        result = engine.route("What is the capital of France?")
        assert result.model_used == "llama3.2:3b"

    def test_result_contains_response(self):
        engine = RouterEngine()
        result = engine.route("What is 2 + 2?")
        assert len(result.response) > 0
