"""
Central configuration for the ComfyUI LLM node.

Holds the shared constants and endpoint handling for the OpenAI-compatible
client. There is no built-in provider -- all requests target a user-supplied
``base_url``.
"""

# --- Default endpoint paths ------------------------------------------------
# The base_url is provided by the user (required). Models list and chat
# completion paths are derived from it. Several OpenAI-compatible servers
# expose /models either at the root or under /v1, so we try both.
MODELS_PATH_CANDIDATES = ("/models", "/v1/models")

# --- Timeouts --------------------------------------------------------------
DEFAULT_REQUEST_TIMEOUT = 120
MIN_REQUEST_TIMEOUT = 1
MAX_REQUEST_TIMEOUT = 3600

# --- Retry -----------------------------------------------------------------
# Number of attempts (>=1) for a single chat completion against transient
# transport errors (connection/TLS/timeout). HTTP 4xx/5xx are not retried.
REQUEST_MAX_RETRIES = 3

# --- Reasoning effort ------------------------------------------------------
REASONING_EFFORT_OPTIONS = ("auto", "none", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT = "auto"

# --- Temperature -----------------------------------------------------------
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
DEFAULT_TEMPERATURE = 1.0

# --- Sampling / generation -------------------------------------------------
# -1 sentinels mean "do not send the field to the endpoint".
DEFAULT_MAX_TOKENS = -1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0

# --- Context window --------------------------------------------------------
# -1 means unlimited: no truncation of prompts sent to the endpoint.
DEFAULT_MAX_CONTEXT_TOKENS = -1
# Rough heuristic used for truncation: tokens-per-character estimate.
CONTEXT_TOKENS_PER_CHAR = 0.25
