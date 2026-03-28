"""
Configurations for the Poison Context Haiku.

Stores model definitions, API keys, and helper functions.
API keys are hardcoded; 
idea: iin the future sue it in the web-app (ES).
"""

from dataclasses import dataclass
from typing import Iterator
from data_loader import load_keys
from typing import Optional

@dataclass
class ModelConfig:
    """Definition of a single LLM endpoint."""
    name: str                # acrtual name (GPT-5)
    litellm_model_id: str    # LiteLLM routing name (openai/gpt-5)
    api_key: str             # API key for this provider
    provider: str            # Provider label (openai, anthropic, gemini, etc.)
    api_base: Optional[str] = None      # Optional custom API base URL (for models like Qwen that don't use the standard OpenAI endpoint)
    enable_thinking: Optional[bool] = None  # Optional flag for models that support "thinking"


# ---------------------------------------------------------------------------
# Commercial models
# ---------------------------------------------------------------------------

COMMERCIAL_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="GPT-5",
        litellm_model_id="openai/gpt-5.2",
        api_key="",
        provider="openai",
    ),
    ModelConfig(
        name="Claude 4",
        litellm_model_id="anthropic/claude-haiku-4-5",      #ES: other option: claude-sonnet-4-5 (more expensive), claude-opus-4-6 (crazy expensive))
        api_key="",
        provider="anthropic",
    ),
    ModelConfig(
        name="Gemini 3",
        litellm_model_id="gemini/gemini-3-flash-preview",   #ES: other option: gemini-3-pro-preview (expensive)
        api_key="",
        provider="gemini",
    ),
    ModelConfig(
        name="Grok",
        litellm_model_id="xai/grok-4-1-fast-non-reasoning",
        api_key="",
        provider="xai",
    ),
]

# ---------------------------------------------------------------------------
# Open models  (TODO: how to connect to the open models?)
# ---------------------------------------------------------------------------

OPEN_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="Llama 4",
        litellm_model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",      # QUESTION: what is {provider}?
        api_key="",
        provider="groq",
    ),
    ModelConfig(
        name="Qwen 3",
        litellm_model_id="dashscope/qwen3-32b",              # QUESTION: what is {provider}?
        api_key="",
        provider="dashscope",
        api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        enable_thinking=False,  
    ),
    ModelConfig(
        name="DeepSeek-R1",
        litellm_model_id="deepseek/deepseek-reasoner",  # QUESTION: what is {provider}?
        api_key="",
        provider="deepseek",
    ),
    ModelConfig(
        name="Mistral",
        litellm_model_id="mistral/mistral-large-latest",  # QUESTION: what is {provider}?
        api_key="",
        provider="mistral",
    ),
]

#Mistral Large 3 (v25.12):

# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

def get_commercial_models() -> list[ModelConfig]:
    """Return all commercial model configs."""
    keys = load_keys()
    if keys:
      for model in COMMERCIAL_MODELS:
        model.api_key = keys[model.provider]
    return list(COMMERCIAL_MODELS)


def get_open_models() -> list[ModelConfig]:
    """Return all open model configs."""
    keys = load_keys()
    if keys:
      for model in OPEN_MODELS:
        model.api_key = keys[model.provider]
    return list(OPEN_MODELS)


def get_all_models() -> list[ModelConfig]:
    """Return commercial + open model configs."""
    return get_commercial_models() + get_open_models()


def get_model_by_name(name: str) -> ModelConfig | None:
    """Look up a single model by its human-readable name (case-insensitive)."""
    for m in get_all_models():
        if m.name.lower() == name.lower():
            return m
    return None

def get_commercial_models_gen() -> Iterator[ModelConfig]:
    keys = load_keys()
    if keys:
      for model in COMMERCIAL_MODELS:
        model.api_key = keys[model.provider]
    yield from COMMERCIAL_MODELS

def get_open_models_gen() -> Iterator[ModelConfig]:
    keys = load_keys()
    if keys:
      for model in OPEN_MODELS:
        model.api_key = keys[model.provider]
    yield from OPEN_MODELS

def get_all_models_gen() -> Iterator[ModelConfig]:
    keys = load_keys()
    ALL_MODELS = COMMERCIAL_MODELS + OPEN_MODELS
    if keys:
      for model in ALL_MODELS:
        model.api_key = keys[model.provider]
    yield from ALL_MODELS
# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Commercial models:")
    for m in get_commercial_models():
        print(f"  {m.name:20s}  {m.litellm_model_id}")

    print("\nOpen models:")
    for m in get_open_models():
        print(f"  {m.name:20s}  {m.litellm_model_id}")

    print(f"\nTotal models: {len(get_all_models())}")

    print("Commercial models iterator:")
    for m in get_commercial_models_gen():
        print(f"  {m.name:20s}  {m.litellm_model_id}") # {m.api_key}")