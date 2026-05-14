"""
RouterEngine — orchestrates the full classify → select model → run flow.

Phase 2 upgrade: LLM-based classification with fallback and confidence escalation.

Classification strategy (in order):
1. LLMClassifier (llama3.2:3b) — smarter, handles ambiguous prompts
2. RuleBasedClassifier (fallback) — used if LLM is unavailable or fails
3. Confidence escalation — if confidence < threshold, override to capable model

This means every routing decision is transparent: the RoutingResult tells you
which classifier was used, which model ran, and why.

Usage:
    engine = RouterEngine()
    result = engine.route("Write a binary search in Python")
    print(result.response)
    print(result.model_used)        # "codellama:7b"
    print(result.routing_reason)    # "[LLM] Code generation task detected"
    print(result.classifier_used)   # "llm" or "rule_based"
"""

import time
from dataclasses import dataclass, field

from router.classifier import RuleBasedClassifier, ClassificationResult
from router.llm_classifier import LLMClassifier, LLMClassifierError, CONFIDENCE_THRESHOLD
from router.registry import ModelRegistry
from models.ollama_client import OllamaClient

# Model used when confidence is too low to trust the classification.
# This is the most capable general-purpose model in the registry.
ESCALATION_MODEL = "llama3.1:8b"


@dataclass
class RoutingResult:
    """
    Full output of a routing request — response plus all routing metadata.

    The metadata fields let callers (and the Phase 3 API) understand exactly
    what happened: which classifier ran, which model was chosen, why, and
    whether escalation occurred.
    """
    prompt: str
    response: str
    model_used: str
    task_type: str
    confidence: float
    reason: str
    elapsed_seconds: float
    model_description: str = ""
    # Phase 2 additions
    classifier_used: str = "rule_based"   # "llm" | "rule_based"
    escalated: bool = False               # True if confidence was too low and model was upgraded

    def __str__(self) -> str:
        escalation_note = " (escalated — low confidence)" if self.escalated else ""
        classifier_note = f"Classifier:  {self.classifier_used}\n"
        return (
            f"Model:      {self.model_used}{escalation_note}\n"
            f"Task type:  {self.task_type} (confidence: {self.confidence:.0%})\n"
            f"Reason:     {self.reason}\n"
            f"{classifier_note}"
            f"Time:       {self.elapsed_seconds:.2f}s\n"
            f"\nResponse:\n{self.response}"
        )


class RouterEngine:
    """
    Main routing engine — wires together classifier, registry, and Ollama client.

    Phase 2 behavior:
    - Tries LLMClassifier first (smarter, handles ambiguous prompts)
    - Falls back to RuleBasedClassifier if LLM is unavailable or errors
    - Escalates to ESCALATION_MODEL if classification confidence is below threshold
    - Records which classifier ran and whether escalation occurred in RoutingResult

    Args:
        model_override: Skip classification and always use this model.
        use_llm_classifier: Set to False to force rule-based only (useful for testing).
    """

    def __init__(self, model_override: str = None, use_llm_classifier: bool = True):
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier() if use_llm_classifier else None
        self.registry = ModelRegistry()
        self.client = OllamaClient()
        self.model_override = model_override

    def route(self, prompt: str) -> RoutingResult:
        """
        Classify the prompt, select a model, run it, and return the full result.

        Classification order:
        1. If model_override is set → skip classification entirely
        2. Try LLMClassifier → get task_type + confidence
        3. If LLM fails → fall back to RuleBasedClassifier (logs the fallback)
        4. If confidence < CONFIDENCE_THRESHOLD → escalate to ESCALATION_MODEL

        Args:
            prompt: The user's input prompt

        Returns:
            RoutingResult with response and full routing metadata
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # ── 1. Classification ──────────────────────────────────────────────
        if self.model_override:
            # Skip classification — user forced a specific model
            classification = ClassificationResult(
                task_type="default",
                confidence=1.0,
                reason=f"Model override: {self.model_override}",
            )
            model_name = self.model_override
            classifier_used = "override"
            escalated = False

        else:
            classification, classifier_used = self._classify(prompt)
            model_name = self.registry.get_model_for(classification.task_type)

            # ── 2. Confidence escalation ───────────────────────────────────
            # If the classifier isn't confident enough, route to the most capable
            # model rather than risk a poor response from a smaller model.
            escalated = classification.confidence < CONFIDENCE_THRESHOLD
            if escalated:
                print(
                    f"  → Low confidence ({classification.confidence:.0%}) — "
                    f"escalating to {ESCALATION_MODEL}"
                )
                model_name = ESCALATION_MODEL

        # ── 3. Run the selected model ──────────────────────────────────────
        profile = self.registry.get_profile(model_name)
        model_description = profile.description if profile else ""

        print(f"  → {classification}")
        print(f"  → Model: {model_name}  |  Classifier: {classifier_used}")

        start = time.time()
        response = self.client.generate(model=model_name, prompt=prompt)
        elapsed = round(time.time() - start, 2)

        return RoutingResult(
            prompt=prompt,
            response=response,
            model_used=model_name,
            task_type=classification.task_type,
            confidence=classification.confidence,
            reason=classification.reason,
            elapsed_seconds=elapsed,
            model_description=model_description,
            classifier_used=classifier_used,
            escalated=escalated,
        )

    def _classify(self, prompt: str) -> tuple[ClassificationResult, str]:
        """
        Run classification with LLM-first, rule-based fallback strategy.

        Tries the LLMClassifier first. If it raises LLMClassifierError
        (model down, bad JSON, etc.), silently falls back to RuleBasedClassifier
        so routing always succeeds.

        Args:
            prompt: The user's input prompt

        Returns:
            (ClassificationResult, classifier_name) where classifier_name
            is "llm" or "rule_based"
        """
        if self.llm_classifier is not None:
            try:
                result = self.llm_classifier.classify(prompt)
                return result, "llm"
            except LLMClassifierError as e:
                # LLM classifier failed — fall back to rule-based and continue
                print(f"  → LLM classifier failed ({e}), falling back to rule-based")

        # Rule-based fallback — always works, no external dependencies
        result = self.rule_classifier.classify(prompt)
        return result, "rule_based"

    def list_models(self) -> list[dict]:
        """Return all registered model profiles as dicts."""
        return [
            {
                "name": p.name,
                "speed": p.speed,
                "task_types": p.task_types,
                "description": p.description,
            }
            for p in self.registry.list_profiles()
        ]
