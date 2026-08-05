"""
A minimal, openai-compatible HTTP client used by all LLM nodes.

Unlike a provider-locked plugin, this client has NO built-in provider. Every
request targets a user-provided ``base_url`` pointing at any OpenAI-compatible
chat/completion server (e.g. a local vLLM/Ollama gateway, a private proxy, or
any vendor exposing an OpenAI-style REST API).
"""

import json
import time
from urllib.parse import urljoin

import requests

from . import config


class LLMClient:
    """
    Wraps the OpenAI-compatible Chat Completions API.

    Attributes:
        base_url:  User-provided endpoint root (required, non-empty).
        api_key:   Optional bearer token for the endpoint.
        timeout:   Request timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: int = None):
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required. Please provide an OpenAI-compatible endpoint.")
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = self.validate_timeout(timeout)

    # ------------------------------------------------------------------ #
    # Validation helpers                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_timeout(timeout):
        try:
            t = int(timeout)
            return max(config.MIN_REQUEST_TIMEOUT, min(config.MAX_REQUEST_TIMEOUT, t))
        except (ValueError, TypeError):
            return config.DEFAULT_REQUEST_TIMEOUT

    @staticmethod
    def validate_temperature(temperature):
        try:
            t = float(temperature)
            return max(config.TEMPERATURE_MIN, min(config.TEMPERATURE_MAX, t))
        except (ValueError, TypeError):
            return config.DEFAULT_TEMPERATURE

    @classmethod
    def validate_reasoning_effort(cls, reasoning_effort):
        if isinstance(reasoning_effort, str):
            normalized = reasoning_effort.strip().lower()
            if normalized in config.REASONING_EFFORT_OPTIONS:
                return normalized
        return config.DEFAULT_REASONING_EFFORT

    # ------------------------------------------------------------------ #
    # Request plumbing                                                    #
    # ------------------------------------------------------------------ #
    def _headers(self):
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_path(self, candidates):
        """
        Return the first candidate path that responds with HTTP 200.
        Builds each candidate against base_url.
        """
        for candidate in candidates:
            url = urljoin(self.base_url + "/", candidate.lstrip("/"))
            try:
                resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
                if resp.status_code == 200:
                    return url
            except requests.exceptions.RequestException:
                continue
        # Return the first candidate URL even if it didn't respond; the
        # caller can surface a meaningful error.
        return urljoin(self.base_url + "/", candidates[0].lstrip("/"))

    def fetch_models(self):
        """
        Fetch the list of model IDs from the endpoint.
        Tries /models and /v1/models and falls back to an empty list on error.
        Returns a sorted list of model id strings.
        """
        url = self._resolve_path(config.MODELS_PATH_CANDIDATES)
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if not isinstance(models, list):
                return []
            ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
            return sorted(ids)
        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            print(f"Error fetching models from {url}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Main chat completion                                                #
    # ------------------------------------------------------------------ #
    def chat_completion(
        self,
        messages,
        model,
        temperature=config.DEFAULT_TEMPERATURE,
        seed=0,
        reasoning_effort=None,
        extra_params=None,
    ):
        if not model or not str(model).strip():
            raise ValueError("model is required. Please enter a model id for your endpoint.")
        """
        Perform a chat completion request.

        Returns a dict containing:
            text          - the text content of the reply
            images        - list of raw base64 image strings (if any)
            response_ms   - server-reported latency (if any)
            usage         - usage dict (if any)
            model         - the model actually used

        Raises requests.exceptions.RequestException on transport errors.
        """
        # Normalize stored message dicts (which may carry extra metadata such as
        # timestamp/session_key/summary) down to only the fields the API accepts.
        cleaned_messages = []
        for m in messages:
            if not isinstance(m, dict):
                cleaned_messages.append(m)
                continue
            clean = {"role": m.get("role", "user")}
            content = m.get("content")
            # Preserve structured multimodal content when it is a list.
            if isinstance(content, list):
                clean["content"] = content
            else:
                clean["content"] = content if content is not None else ""
            cleaned_messages.append(clean)
        messages = cleaned_messages

        data = {
            "model": model,
            "messages": messages,
            "temperature": self.validate_temperature(temperature),
            "seed": int(seed),
        }

        validated_effort = self.validate_reasoning_effort(reasoning_effort or config.DEFAULT_REASONING_EFFORT)
        if validated_effort != "auto":
            data["reasoning_effort"] = validated_effort

        if extra_params:
            for key, value in extra_params.items():
                if value is not None and value != "":
                    data[key] = value

        url = urljoin(self.base_url + "/", "chat/completions")
        resp = requests.post(url, headers=self._headers(), json=data, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()

        message = (result.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or ""

        # Normalize every image source into a plain base64 data string.
        images = []

        def add_image_data(url_val):
            if isinstance(url_val, str) and url_val.startswith("data:image"):
                return url_val.split(",", 1)[1] if "," in url_val else url_val
            return None

        # Modern format: message["images"] = [{"image_url": {...}}]
        for img in message.get("images") or []:
            if isinstance(img, dict):
                url_val = img.get("image_url") or {}
                if isinstance(url_val, dict):
                    url_val = url_val.get("url")
                image_data = add_image_data(url_val)
            elif isinstance(img, str):
                image_data = add_image_data(img)
            else:
                image_data = None
            if image_data:
                images.append(image_data)

        # Legacy multimodal content: text may be a list of blocks containing
        # image_url blocks.
        if isinstance(text, list):
            parts = []
            for item in text:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    data = add_image_data((item.get("image_url") or {}).get("url"))
                    if data:
                        images.append(data)
            text = "\n".join(parts)

        return {
            "text": text,
            "images": images,
            "response_ms": result.get("response_ms"),
            "usage": result.get("usage", {}),
            "model": result.get("model", model),
        }
