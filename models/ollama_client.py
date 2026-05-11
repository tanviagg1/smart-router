"""
OllamaClient — thin wrapper around the ollama Python SDK.

All model calls in smart-router go through this class.
It handles the chat call and returns a plain string response.
"""

import ollama


class OllamaClient:
    """
    Wraps ollama.Client to provide a simple generate() interface.

    Usage:
        client = OllamaClient()
        response = client.generate("llama3.1:8b", "What is recursion?")
    """

    def __init__(self):
        self.client = ollama.Client()

    def generate(self, model: str, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """
        Send a prompt to the specified model and return the text response.

        Args:
            model: Ollama model name (e.g. "llama3.1:8b")
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature (0=deterministic, 1=creative)

        Returns:
            The model's text response as a string.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response.message.content.strip()

    def is_available(self, model: str) -> bool:
        """Check if a model is available locally in Ollama."""
        try:
            models = self.client.list()
            available = [m.model for m in models.models]
            return any(model in m for m in available)
        except Exception:
            return False
