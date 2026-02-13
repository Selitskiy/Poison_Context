"""
Configurations for the Poison Context Haiku.

Stores model definitions, API keys, and helper functions.
API keys are hardcoded; 
idea: iin the future sue it in the web-app (ES).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Definition of a single LLM endpoint."""
    name: str                # acrtual name (GPT-5)
    litellm_model_id: str    # LiteLLM routing name (openai/gpt-5)
    api_key: str             # API key for this provider
    provider: str            # Provider label (openai, anthropic, gemini, etc.)


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
        litellm_model_id="{provider}/meta-llama/Llama-4",      # QUESTION: what is {provider}?
        api_key="PASTE_OPEN_PROVIDER_API_KEY_HERE",
        provider="TODO",
    ),
    ModelConfig(
        name="Qwen 3",
        litellm_model_id="{provider}/Qwen/Qwen3",              # QUESTION: what is {provider}?
        api_key="PASTE_OPEN_PROVIDER_API_KEY_HERE",
        provider="TODO",
    ),
    ModelConfig(
        name="DeepSeek-R1",
        litellm_model_id="{provider}/deepseek-ai/DeepSeek-R1",  # QUESTION: what is {provider}?
        api_key="PASTE_OPEN_PROVIDER_API_KEY_HERE",
        provider="TODO",
    ),
    ModelConfig(
        name="Mistral",
        litellm_model_id="{provider}/mistralai/Mistral-Large",  # QUESTION: what is {provider}?
        api_key="PASTE_OPEN_PROVIDER_API_KEY_HERE",
        provider="TODO",
    ),
]


# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------

def get_commercial_models() -> list[ModelConfig]:
    """Return all commercial model configs."""
    return list(COMMERCIAL_MODELS)


def get_open_models() -> list[ModelConfig]:
    """Return all open model configs."""
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
