"""
ModelRegistry — maps task types to Ollama models with capability profiles.

Each model has a profile describing what it's good at, its speed tier,
and the task types it handles best. The router uses this to select the
right model for each classified task.

Task types:
  simple_qa  — short factual questions, definitions, quick lookups
  code       — code generation, debugging, algorithms, syntax help
  complex    — deep reasoning, explanations, analysis, long-form content
  default    — fallback for anything unclassified
"""

from dataclasses import dataclass, field


@dataclass
class ModelProfile:
    """Describes a model's capabilities and routing metadata."""
    name: str               # Ollama model name (e.g. "llama3.1:8b")
    speed: str              # "fast" | "medium" | "slow"
    task_types: list[str]   # task types this model handles
    description: str        # human-readable capability summary
    max_tokens: int = 4096  # approximate context window


# ---------------------------------------------------------------------------
# Default model profiles
# ---------------------------------------------------------------------------

DEFAULT_PROFILES = [
    ModelProfile(
        name="llama3.2:3b",
        speed="fast",
        task_types=["simple_qa", "default"],
        description="Fast, lightweight model. Best for short factual Q&A and simple tasks.",
        max_tokens=4096,
    ),
    ModelProfile(
        name="codellama:7b",
        speed="medium",
        task_types=["code"],
        description="Code-specialized model. Best for code generation, debugging, and algorithms.",
        max_tokens=4096,
    ),
    ModelProfile(
        name="llama3.1:8b",
        speed="medium",
        task_types=["complex", "simple_qa", "code", "default"],
        description="General-purpose capable model. Handles complex reasoning and long-form content.",
        max_tokens=8192,
    ),
]


class ModelRegistry:
    """
    Stores model profiles and resolves which model to use for a given task type.

    Usage:
        registry = ModelRegistry()
        model_name = registry.get_model_for("code")  # → "codellama:7b"
    """

    def __init__(self, profiles: list[ModelProfile] = None):
        self._profiles: dict[str, ModelProfile] = {}
        for profile in (profiles or DEFAULT_PROFILES):
            self._profiles[profile.name] = profile

    def get_model_for(self, task_type: str) -> str:
        """
        Return the best model name for a given task type.

        Selection priority:
        1. A model whose primary task_types[0] matches
        2. A model that lists the task type anywhere in task_types
        3. The default fallback model (llama3.1:8b)

        Args:
            task_type: One of simple_qa, code, complex, default

        Returns:
            Ollama model name string
        """
        # Primary match: model whose first task type matches
        for profile in self._profiles.values():
            if profile.task_types and profile.task_types[0] == task_type:
                return profile.name

        # Secondary match: model that supports this task type at all
        for profile in self._profiles.values():
            if task_type in profile.task_types:
                return profile.name

        # Fallback
        return "llama3.1:8b"

    def get_profile(self, model_name: str) -> ModelProfile | None:
        """Return the profile for a given model name."""
        return self._profiles.get(model_name)

    def list_profiles(self) -> list[ModelProfile]:
        """Return all registered model profiles."""
        return list(self._profiles.values())

    def register(self, profile: ModelProfile) -> None:
        """Add or replace a model profile."""
        self._profiles[profile.name] = profile
