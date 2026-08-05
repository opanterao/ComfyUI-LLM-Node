"""
A minimal, openai-compatible client used by all LLM nodes.

This client wraps the official ``openai`` SDK so it uses the same battle-tested
HTTP/httpx transport that works reliably against OpenAI-compatible servers
(e.g. the CSYIDC API platform), avoiding the intermittent TLS failures seen with
hand-rolled ``requests`` POSTs of large image payloads.

There is NO built-in provider. Every request targets a user-provided
``base_url`` pointing at any OpenAI-compatible chat/completion server (a local
vLLM/Ollama gateway, a private proxy, or any vendor exposing an OpenAI-style
REST API).
"""

import json
from urllib.parse import urljoin

import requests

from openai import OpenAI

from . import config


class LLMClient:
    """
    Wraps the OpenAI-compatible Chat Completions API via the openai SDK.

    Attributes:
        base_url:  User-provided endpoint root (required, non-empty).
        api_key:   Optional bearer token for the endpoint.
        timeout:   Request timeout in seconds.
    """

    # Parameters the openai SDK accepts directly as named arguments.
    _NATIVE_PARAMS = {
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "max_completion_tokens",
    }

    def __init__(self, base_url: str, api_key: str = "", timeout: int = None):
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required. Please provide an OpenAI-compatible endpoint.")
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = self.validate_timeout(timeout)
        # openai SDK requires a non-empty api_key even for auth-less endpoints,
        # so fall back to a harmless placeholder.
        self._client = OpenAI(
            api_key=self.api_key or "sk-none",
            base_url=self.base_url,
            timeout=self.timeout,
        )

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
    # Model list (still uses a lightweight requests GET)                  #
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
    # Main chat completion (via openai SDK)                               #
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
        Perform a chat completion request via the openai SDK.

        Returns a dict containing:
            text   - the text content of the reply
            usage  - usage dict (if any)
            model  - the model actually used

        Raises openai.APIError subclasses (e.g. APIConnectionError, APIStatusError)
        on transport/timeout errors and bad HTTP responses.
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

        create_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.validate_temperature(temperature),
            "seed": int(seed),
        }

        validated_effort = self.validate_reasoning_effort(
            reasoning_effort or config.DEFAULT_REASONING_EFFORT
        )
        if validated_effort != "auto":
            create_kwargs["reasoning_effort"] = validated_effort

        # Split extra params into SDK-native named args vs. extra_body for
        # non-standard fields (e.g. top_k).
        extra_body = {}
        if extra_params:
            for key, value in extra_params.items():
                if value is None or value == "":
                    continue
                if key in self._NATIVE_PARAMS:
                    create_kwargs[key] = value
                else:
                    extra_body[key] = value
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        result = self._client.chat.completions.create(**create_kwargs)

        message = result.choices[0].message if result.choices else None
        text = ""
        if message is not None:
            if isinstance(message.content, list):
                text = "\n".join(
                    str(item.get("text", ""))
                    for item in message.content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            elif message.content is not None:
                text = str(message.content)

        usage = result.usage
        usage_dict = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return {
            "text": text,
            "usage": usage_dict,
            "model": getattr(result, "model", None) or model,
        }
