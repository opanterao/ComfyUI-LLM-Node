"""
VLM multi-reference-image description node.

Describes up to ``MAX_IMAGES`` reference images with a vision-language model
and merges the descriptions into one text output, all against a user-supplied
OpenAI-compatible endpoint (no built-in provider).
"""

import torch

from . import config
from .api_client import LLMClient
from .node import _input_keys, image_to_base64


def _placeholder_image():
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _truncate_text(text: str, max_context_tokens: int) -> str:
    """Cap ``text`` to roughly ``max_context_tokens`` tokens using a heuristic.

    Used to avoid overflowing the endpoint's context window. ``-1`` means no
    truncation.
    """
    if max_context_tokens is None or max_context_tokens < 0 or not text:
        return text
    max_chars = int(max_context_tokens / config.CONTEXT_TOKENS_PER_CHAR)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[...truncated]"


class CSYIDCRefDescribeNode:
    """
    Multi-reference-image VLM description node.

    Accepts a variable number of ``ref_image_N`` inputs (1..MAX_IMAGES). Each
    image is described independently by the endpoint's vision-language model
    using the ``image_instruction`` prompt. All descriptions are then combined
    and passed back to the model together with the user's ``system_prompt``,
    which dictates how to merge them into one complete text output.
    """

    MAX_IMAGES = 15

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Output",)
    FUNCTION = "describe"
    CATEGORY = "LLM"

    reasoning_effort_options = config.REASONING_EFFORT_OPTIONS
    default_reasoning_effort = config.DEFAULT_REASONING_EFFORT

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "你得到了多张参考图片的描述。请按照用户的指令，"
                        "将它们合并成一段完整、连贯的回复。"
                    ),
                }),
                "image_instruction": ("STRING", {
                    "multiline": True,
                    "default": (
                        "请详细描述这张图片，包括构图、主体、光线、色彩和风格。"
                    ),
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "your-model-id",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": config.TEMPERATURE_MIN, "max": config.TEMPERATURE_MAX, "step": 0.01, "display": "slider"}),
                "reasoning_effort": (
                    list(cls.reasoning_effort_options),
                    {"default": cls.default_reasoning_effort},
                ),
                "max_tokens": ("INT", {
                    "default": config.DEFAULT_MAX_TOKENS, "min": -1, "max": 1048576, "step": 1,
                    "display": "number",
                }),
                "max_context_tokens": ("INT", {
                    "default": config.DEFAULT_MAX_CONTEXT_TOKENS, "min": -1, "max": 1048576, "step": 1,
                    "display": "number",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": "randomize",
                }),
                "top_p": ("FLOAT", {"default": config.DEFAULT_TOP_P, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "top_k": ("INT", {"default": config.DEFAULT_TOP_K, "min": -1, "max": 1000, "step": 1, "display": "number"}),
                "frequency_penalty": ("FLOAT", {"default": config.DEFAULT_FREQUENCY_PENALTY, "min": -2.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "presence_penalty": ("FLOAT", {"default": config.DEFAULT_PRESENCE_PENALTY, "min": -2.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "stop": ("STRING", {"multiline": True, "default": "", "placeholder": "Comma-separated stop sequences (optional)"}),
                "request_timeout": ("INT", {"default": 300, "min": 1, "max": 3600, "step": 1}),
                "max_image_size": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "user_instruction": ("STRING", {"forceInput": True, "default": ""}),
            },
        }

    def _build_client(self, base_url, api_key, request_timeout):
        timeout = LLMClient.validate_timeout(request_timeout)
        try:
            return LLMClient(base_url, api_key, timeout)
        except ValueError as e:
            return str(e)

    @staticmethod
    def _build_extra_params(max_tokens, top_p, top_k, frequency_penalty, presence_penalty, stop):
        """Assemble optional completion params, omitting -1 sentinels and empties."""
        params = {}

        def put_nonneg(name, value):
            if isinstance(value, (int, float)) and value < 0:
                return
            params[name] = value

        put_nonneg("max_tokens", max_tokens)
        put_nonneg("top_k", top_k)
        params["top_p"] = top_p
        params["frequency_penalty"] = frequency_penalty
        params["presence_penalty"] = presence_penalty

        if isinstance(stop, str) and stop.strip():
            seqs = [s.strip() for s in stop.split(",") if s.strip()]
            if seqs:
                params["stop"] = seqs
        return params

    def _describe_image(self, client, model, temperature, seed, reasoning_effort,
                        instruction, image_tensor, extra_params, max_image_size=1024):
        """Run the VLM once on a single image and return its description text."""
        b64 = image_to_base64(image_tensor, max_size=max_image_size)
        messages = [
            {"role": "system", "content": "You are a careful image analyst."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]
        result = client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            reasoning_effort=reasoning_effort,
            extra_params=extra_params,
        )
        return (result.get("text") or "").strip()

    def describe(
        self,
        base_url,
        api_key,
        system_prompt,
        image_instruction,
        model,
        temperature,
        reasoning_effort,
        max_tokens,
        max_context_tokens,
        seed,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        stop,
        request_timeout,
        **kwargs,
    ):
        client = self._build_client(base_url, api_key, request_timeout)
        if isinstance(client, str):
            return (f"Error: {client}",)

        user_instruction = kwargs.get("user_instruction") or ""
        max_image_size = kwargs.get("max_image_size")

        images = []
        for key in _input_keys(kwargs, "ref_image"):
            if kwargs.get(key) is None:
                continue
            images.append((key, kwargs[key]))

        if not images:
            return ("Error: at least one reference image is required.",)

        if len(images) > self.MAX_IMAGES:
            return (f"Error: too many images ({len(images)}). Maximum is {self.MAX_IMAGES}.",)

        extra_params = self._build_extra_params(
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        # 1) Describe each image independently.
        per_image_instruction = _truncate_text(image_instruction, max_context_tokens)
        descriptions = []
        for i, (key, tensor) in enumerate(images, start=1):
            try:
                desc = self._describe_image(
                    client, model, temperature, seed, reasoning_effort,
                    per_image_instruction, tensor, extra_params,
                    max_image_size=max_image_size,
                )
            except Exception as e:
                return (f"Error describing {key}: {e}",)
            if not desc:
                desc = "(no description returned)"
            descriptions.append(f"[Image {i} ({key})]\n{desc}")

        combined = "\n\n".join(descriptions)

        # 2) Combine all descriptions into one complete output per the system prompt.
        # Always pass the per-image descriptions so the model can use them; the
        # user instruction (if any) is merged in to frame the final output.
        if user_instruction.strip():
            user_text = (
                f"The following are descriptions of reference images:\n\n{combined}\n\n"
                f"User instruction:\n{user_instruction.strip()}\n\n"
                "Produce the final output according to the system prompt."
            )
        else:
            user_text = (
                f"The following are descriptions of reference images:\n\n{combined}\n\n"
                "Produce the final output according to the system prompt."
            )

        user_text = _truncate_text(user_text, max_context_tokens)
        combine_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            result = client.chat_completion(
                messages=combine_messages,
                model=model,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
                extra_params=extra_params,
            )
        except Exception as e:
            return (f"API Request Error: {e}",)

        return (result.get("text") or "",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "CSYIDC-RefDescribe": CSYIDCRefDescribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSYIDC-RefDescribe": "CSYIDC Multi-Reference Image Describe",
}
