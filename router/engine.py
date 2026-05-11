"""
RouterEngine — orchestrates the full classify → select model → run flow.

This is the main entry point for routing a prompt. It:
1. Classifies the prompt using RuleBasedClassifier
2. Looks up the right model from ModelRegistry
3. Calls the model via OllamaClient
4. Returns a RoutingResult with the response + full routing metadata

Usage:
    engine = RouterEngine()
    result = engine.route("Write a binary search in Python")
    print(result.response)
    print(result.model_used)   # "codellama:7b"
    print(result.routing.reason)
"""

import time
from dataclasses import dataclass, field

from router.classifier import RuleBasedClassifier, ClassificationResult
from router.registry import ModelRegistry
from models.ollama_client import OllamaClient


@dataclass
class RoutingResult:
    """
    The full output of a routing request.

    Contains the model response plus all routing metadata so callers
    can see exactly why a particular model was chosen.
    """
    prompt: str
    response: str
    model_used: str
    task_type: str
    confidence: float
    reason: str
    elapsed_seconds: float
    model_description: str = ""

    def __str__(self) -> str:
        return (
            f"Model:      {self.model_used}\n"
            f"Task type:  {self.task_type} (confidence: {self.confidence:.0%})\n"
            f"Reason:     {self.reason}\n"
            f"Time:       {self.elapsed_seconds:.2f}s\n"
            f"\nResponse:\n{self.response}"
        )


class RouterEngine:
    """
    Main routing engine.

    Wires together the classifier, registry, and Ollama client.
    Every call to route() returns a RoutingResult with full transparency
    into why a particular model was selected.

    Args:
        model_override: If set, skip classification and always use this model.
    """

    def __init__(self, model_override: str = None):
        self.classifier = RuleBasedClassifier()
        self.registry = ModelRegistry()
        self.client = OllamaClient()
        self.model_override = model_override

    def route(self, prompt: str) -> RoutingResult:
        """
        Classify the prompt, select a model, run it, and return the result.

        Args:
            prompt: The user's input prompt

        Returns:
            RoutingResult with response + full routing metadata
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # 1. Classify
        if self.model_override:
            classification = ClassificationResult(
                task_type="default",
                confidence=1.0,
                reason=f"Model override: {self.model_override}",
            )
            model_name = self.model_override
        else:
            classification = self.classifier.classify(prompt)
            model_name = self.registry.get_model_for(classification.task_type)

        # 2. Get profile for description
        profile = self.registry.get_profile(model_name)
        model_description = profile.description if profile else ""

        # 3. Run model
        print(f"  → {classification}")
        print(f"  → Model: {model_name}")

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
        )

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
