"""
LLMClassifier — uses llama3.2:3b to classify prompt task type.

Why LLM-based classification?
  Rule-based classifiers miss ambiguous prompts — things like
  "walk me through setting up a Flask app" hit both code and complex signals.
  A small LLM understands intent and phrasing better without being expensive.
  llama3.2:3b is fast and cheap, making it suitable as a routing step.

How it works:
  1. Sends the prompt to llama3.2:3b with a structured system instruction.
  2. Asks the model to respond with JSON: task_type, confidence, reason.
  3. Parses the JSON and returns a ClassificationResult.
  4. On any failure (model unavailable, bad JSON, timeout) → raises
     LLMClassifierError so the caller can fall back gracefully.

Confidence threshold:
  If confidence < CONFIDENCE_THRESHOLD (0.65), the result is uncertain.
  RouterEngine uses this to escalate routing to a larger, more capable model
  rather than blindly trusting the classification.

Usage:
    clf = LLMClassifier()
    result = clf.classify("Walk me through setting up a Flask app")
    # → ClassificationResult(task_type="code", confidence=0.85, reason="...")

    # Check if model is available before using
    if clf.is_available():
        result = clf.classify(prompt)
"""

import json
import re

from models.ollama_client import OllamaClient
from router.classifier import ClassificationResult

# The model used for classification — small and fast to keep routing overhead low
CLASSIFIER_MODEL = "llama3.2:3b"

# If the LLM returns confidence below this threshold, RouterEngine will
# escalate to a larger model to compensate for the uncertain classification
CONFIDENCE_THRESHOLD = 0.65

# System prompt that instructs the LLM to act as a classifier.
# Temperature is set to 0 for deterministic, consistent output.
# We ask for strict JSON so the response is easy to parse.
SYSTEM_PROMPT = """You are a prompt classifier for an LLM routing system.
Your job is to classify a user prompt into exactly one task type.

Task types:
- simple_qa   : Short factual questions, definitions, quick lookups (e.g. "What is recursion?")
- code        : Code generation, debugging, algorithms, scripts, data structures (e.g. "Write a binary search")
- complex     : Deep reasoning, explanations, comparisons, architecture, analysis (e.g. "Explain CAP theorem in depth")
- default     : Anything that doesn't clearly fit the above

Rules:
- If the prompt asks to write, implement, debug, or fix code → code
- If the prompt asks for a definition, fact, or short answer → simple_qa
- If the prompt asks to explain, compare, analyze, or discuss in detail → complex
- When unsure, pick the closest match and lower your confidence score

Respond with ONLY valid JSON. No explanation outside the JSON block.
Format:
{
  "task_type": "<simple_qa|code|complex|default>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the classification>"
}"""


class LLMClassifierError(Exception):
    """
    Raised when the LLM classifier fails for any reason:
    - Model not available in Ollama
    - Response couldn't be parsed as valid JSON
    - Unexpected response format
    RouterEngine catches this and falls back to RuleBasedClassifier.
    """
    pass


class LLMClassifier:
    """
    Classifies a prompt by asking llama3.2:3b to reason about its task type.

    This is Phase 2's primary classifier. It handles ambiguous prompts that
    rule-based matching struggles with. The RuleBasedClassifier is kept as a
    fallback in RouterEngine in case this classifier fails.

    Args:
        model: Ollama model name to use for classification (default: llama3.2:3b)
        confidence_threshold: Below this value, RouterEngine escalates the model

    Usage:
        clf = LLMClassifier()
        result = clf.classify("How do I center a div in CSS?")
        print(result.task_type)    # "code"
        print(result.confidence)   # 0.88
        print(result.reason)       # "Asking about CSS implementation..."
    """

    def __init__(self, model: str = CLASSIFIER_MODEL):
        self.model = model
        self.client = OllamaClient()

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify the prompt using the LLM.

        Sends the prompt to llama3.2:3b with a structured system instruction
        and parses the JSON response into a ClassificationResult.

        Args:
            prompt: The user's input prompt

        Returns:
            ClassificationResult with task_type, confidence, and reason

        Raises:
            LLMClassifierError: If the model is unavailable or returns unparseable output
        """
        # Ask the LLM to classify — temperature=0 for deterministic output
        try:
            raw = self.client.generate(
                model=self.model,
                prompt=f'Classify this prompt:\n\n"{prompt}"',
                system=SYSTEM_PROMPT,
                temperature=0.0,
            )
        except Exception as e:
            raise LLMClassifierError(f"LLM call failed: {e}") from e

        # Parse the JSON response from the model
        return self._parse_response(raw, prompt)

    def _parse_response(self, raw: str, original_prompt: str) -> ClassificationResult:
        """
        Parse the LLM's raw text output into a ClassificationResult.

        The LLM is instructed to return JSON, but may wrap it in markdown
        code fences (```json ... ```) — we strip those before parsing.

        Args:
            raw: Raw text response from the LLM
            original_prompt: Original prompt (used in error messages)

        Returns:
            ClassificationResult

        Raises:
            LLMClassifierError: If JSON is missing, malformed, or has invalid values
        """
        # Strip markdown code fences if the model wrapped the JSON
        # e.g. ```json\n{...}\n``` → {...}
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        # Find the first JSON object in the response (in case of preamble text)
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            raise LLMClassifierError(
                f"No JSON found in LLM response. Raw output: {raw[:200]}"
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise LLMClassifierError(
                f"Failed to parse LLM JSON response: {e}. Raw: {raw[:200]}"
            ) from e

        # Validate required fields
        task_type = data.get("task_type", "").strip()
        valid_types = {"simple_qa", "code", "complex", "default"}
        if task_type not in valid_types:
            raise LLMClassifierError(
                f"LLM returned unknown task_type '{task_type}'. Expected one of {valid_types}"
            )

        # Clamp confidence to [0.0, 1.0] in case the model returns out-of-range values
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5  # safe default if confidence is malformed

        reason = str(data.get("reason", "LLM classification")).strip()

        return ClassificationResult(
            task_type=task_type,
            confidence=confidence,
            reason=f"[LLM] {reason}",
            matched_signals=[],
        )

    def is_available(self) -> bool:
        """Check if the classifier model is available in Ollama."""
        return self.client.is_available(self.model)
