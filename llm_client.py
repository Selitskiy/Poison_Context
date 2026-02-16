"""LLM client wrapper around LiteLLM: single-turn and multi-turn helpers with retry."""
"""temeprature notes:
0.0: Always picks the most likely token (deterministic). Same input → same output every time.
0.2 - 0.5: Mostly predictable with slight variation
0.7: Balanced creativity vs coherence
1.0: Provider default. More diverse, occasionally surprising (Creative writing)
1.5 - 2.0: Very random
"""

import time

import litellm

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2a

litellm.suppress_debug_info = True
litellm.drop_params = True  #ES: This tells LiteLLM to automatically drop any parameters that a specific model doesn't support (like temperature for GPT-5)

MAX_RETRIES: int = 3
INITIAL_BACKOFF_SECONDS: float = 2.0  # doubles each retry: 2 -> 4 -> 8


def _call_with_retries(
    model_config: ModelConfig,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Call litellm.completion with retry + exponential back-off."""
    last_exception: Exception | None = None
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            kwargs: dict = dict(
                model=model_config.litellm_model_id,
                messages=messages,
                temperature=temperature,        # ES: better to have this option for experimentation with different temperatures
                api_key=model_config.api_key,   # ES:better to use this option instead of the environment variable (due tomultiple models)
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = litellm.completion(**kwargs)
            elapsed = time.perf_counter() - t0
            reply: str = response.choices[0].message.content or ""

            print(f"[{model_config.name}] OK in {elapsed:.1f}s (attempt {attempt})")
            return reply

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            last_exception = exc
            print(f"[{model_config.name}] Attempt {attempt}/{MAX_RETRIES} failed after {elapsed:.1f}s: {exc}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {backoff:.1f}s ...")
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(
        f"[{model_config.name}] All {MAX_RETRIES} attempts failed. Last error: {last_exception}"
    ) from last_exception


def single_turn(
    model_config: ModelConfig,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Send a one-shot user prompt and return the assistant's reply."""
    messages = [{"role": "user", "content": prompt}]
    return _call_with_retries(model_config, messages, temperature=temperature, max_tokens=max_tokens)


def multi_turn(
    model_config: ModelConfig,
    messages: list[dict],
    *,
    temperature: float = 0.7,   #ES: Balanced creativity vs coherence (0.0 = deterministic, 1.0 = most creative)
    max_tokens: int | None = None,
) -> str:
    """Send a multi-turn conversation and return the assistant's latest reply.
    Used for poison-injection flow (prompt 2a + 2b in one conversation).
    """
    return _call_with_retries(model_config, messages, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    mConfigIter = get_commercial_models_gen()
    mConf = next(mConfigIter)
    mConf = next(mConfigIter)
    mConf = next(mConfigIter)
    #mConf = next(mConfigIter)
    if mConf.api_key: 
      haikuList = load_haiku()
      haiku = haikuList[0]

      #prompt = prompt_1(haiku.haiku)
      prompt = prompt_2a(haiku.haiku, haiku.injection)
      answer = single_turn(mConf, prompt)
      print(f"{answer}")
    else:
      print(f"API KEY is empty for {mConf.name}")