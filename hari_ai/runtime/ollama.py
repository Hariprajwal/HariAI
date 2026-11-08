"""Small, dependency-free client for a locally running Ollama service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class OllamaClient:
    """Selects an installed local model and sends chat requests to Ollama."""

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 120.0, discovery_timeout: float = 3.0):
        self.endpoint = (endpoint or os.environ.get(
            "HARIAI_OLLAMA_HOST", "http://127.0.0.1:11434"
        )).rstrip("/")
        self.preferred_model = model or os.environ.get("HARIAI_OLLAMA_MODEL")
        self.timeout = timeout
        self.discovery_timeout = discovery_timeout
        self.last_error: Optional[str] = None

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.endpoint}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.discovery_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                self.last_error = None
                return result
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            self.last_error = str(exc)
            return None

    def list_models(self) -> List[str]:
        """Return installed model names, or an empty list if Ollama is offline."""
        result = self._request("/api/tags")
        if not result:
            return []
        return [str(item["name"]) for item in result.get("models", []) if item.get("name")]

    def select_model(self, available: Optional[List[str]] = None) -> Optional[str]:
        """Use an explicitly configured model, otherwise choose a capable installed model."""
        models = available if available is not None else self.list_models()
        if self.preferred_model:
            return next((name for name in models if name == self.preferred_model), None)
        if not models:
            return None

        # Favor general instruction models; avoid selecting an embedding-only model.
        ranked = sorted(models, key=lambda name: (
            "embed" in name.lower() or "rerank" in name.lower(),
            not any(key in name.lower() for key in (
                "qwen", "llama", "gemma", "mistral", "phi", "deepseek", "command"
            )),
            name.lower(),
        ))
        return ranked[0]

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        """Generate a response from the selected local model; return None on failure."""
        model = self.select_model()
        if model is None:
            if self.preferred_model:
                self.last_error = f"Configured Ollama model '{self.preferred_model}' is not installed or Ollama is offline."
            return None
        result = self._request("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }, timeout=self.timeout)
        if not result:
            return None
        message = result.get("message") or {}
        content = message.get("content", "")
        return content.strip() if isinstance(content, str) and content.strip() else None

