"""
Generation runner for the Poison Context Haiku experiment (Step 6).

For each (model, haiku_entry) pair:
  1. Ablation:  single_turn  prompt_1  --> response_1
  2. Cleanup:   single_turn  prompt_1c (discard)
  3. Poison:    multi_turn   prompt_2a then prompt_2b  --> response_2b
  4. Cleanup:   single_turn  prompt_2c (discard)
  5. Append row to generation_results.csv

Features:
  - Resume support: skip rows already present in the output CSV (matched on model + haiku).
  - Configurable delay between API calls to respect rate limits.
  - Processes one model at a time, one haiku at a time, for easy debugging.

Example:
python run_generation.py --models commercial --haiku-limit 2
    Run only commercial models (GPT-5, Claude 4, Gemini 3, Grok) with the first 2 haiku entries.
python run_generation.py --model-name "GPT-5" --haiku-limit 5
    Run only GPT-5 with the first 5 haiku entries.
python run_generation.py --models all --no-resume --delay 2.0
    Run all models (commercial + open) without resuming and with a 2.0 second delay between API calls.  
"""

import csv
import os
import time
from datetime import datetime, timezone

from config import ModelConfig, get_commercial_models, get_all_models, get_model_by_name
from data_loader import load_haiku, HaikuEntry
from llm_client import single_turn, multi_turn
from prompts import prompt_1, prompt_1c, prompt_2a, prompt_2b, prompt_2c

# Output CSV lives next to this script
_OUTPUT_DIR = os.path.dirname(__file__)
_OUTPUT_CSV = os.path.join(_OUTPUT_DIR, "generation_results.csv")

# Pipe delimiter avoids conflicts with commas inside haiku text
_DELIMITER = "|"

_FIELDNAMES = [
    "model",
    "haiku",
    "translation",
    "injection",
    "response_1",
    "response_2b",
    "timestamp",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_existing_keys(csv_path: str) -> set[tuple[str, str]]:
    """Return a set of (model, haiku) pairs already recorded in the output CSV."""
    keys: set[tuple[str, str]] = set()
    if not os.path.isfile(csv_path):
        return keys

    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=_DELIMITER)    # ES: this is a dictionary reader that will read the CSV file and return a dictionary of the rows
        for row in reader:
            keys.add((row["model"], row["haiku"]))

    return keys


def _append_row(csv_path: str, row: dict) -> None:
    """Append a single row to the CSV, creating the file + header if needed."""
    write_header = not os.path.isfile(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, delimiter=_DELIMITER)    # ES: this is a dictionary writer that will write the row to the CSV file
        if write_header:
            writer.writeheader()    # ES: this will write the header to the CSV file if it doesn't exist
        writer.writerow(row)        # ES: this will write the row to the CSV file (row is a dictionary with the keys and values to write)


# ── core runner ──────────────────────────────────────────────────────────────

def run_generation(
    models: list[ModelConfig],
    entries: list[HaikuEntry],
    *,
    delay: float = 1.0,
    resume: bool = True,
    output_csv: str = _OUTPUT_CSV,
) -> None:
    """
    Execute the generation experiment for every (model, haiku) combination.

    Parameters
    ----------
    models : list[ModelConfig]
        Models to generate with.
    entries : list[HaikuEntry]
        Haiku entries loaded from the CSV.
    delay : float
        Seconds to wait between consecutive API calls (to ste with the API rate limits).
    resume : bool
        If True, skip (model, haiku) pairs that already have a row in the output CSV (useful for resuming the script to avoid re-running the same pairs and potentially getting charged for the same stuff again).
    output_csv : str
        Path to the output CSV file (where the results will be saved).
    """
    # Load already-completed keys for resume support
    existing_keys = _load_existing_keys(output_csv) if resume else set()    # ES: this is a set of (model, haiku) pairs that already have a row in the output CSV
    if existing_keys:
        print(f"Resume mode: {len(existing_keys)} existing row(s) found — will skip them.\n") # ES: this is a nice way to show the user the progress of the script

    total_pairs = len(models) * len(entries) # ES: this is the total number of (model, haiku) pairs that will be processed
    completed = 0
    skipped = 0

    for model in models:
        print(f"\n{'='*72}")
        print(f"MODEL: {model.name}  ({model.litellm_model_id})") # ES: this is a nice way to show the user the model that is being processed
        print(f"{'='*72}")

        for idx, entry in enumerate(entries, start=1):
            key = (model.name, entry.haiku)

            if key in existing_keys:
                skipped += 1
                print(f"  [{idx}/{len(entries)}] SKIP (already exists): {entry.haiku[:40]}...") # ES: this is a nice way to show the user the progress of the script
                continue

            print(f"\n  [{idx}/{len(entries)}] Processing: {entry.haiku[:50]}...") # ES: this is a nice way to show the user the progress of the script

            # ── Step 1: Ablation generation (Prompt 1) ───────────────────
            print("    → Prompt 1 (ablation generation) ...")
            response_1 = single_turn(model, prompt_1(entry.haiku))
            time.sleep(delay)

            # ── Step 2: Cleanup (Prompt 1c) ──────────────────────────────
            print("    → Prompt 1c (cleanup) ...")
            single_turn(model, prompt_1c(entry.haiku))
            time.sleep(delay)

            # ── Step 3: Poison injection + poisoned generation (2a → 2b) ─
            print("    → Prompt 2a (poison injection) ...")
            messages_2a = [{"role": "user", "content": prompt_2a(entry.haiku, entry.injection)}]
            response_2a = multi_turn(model, messages_2a)
            time.sleep(delay)

            print("    → Prompt 2b (poisoned generation, same conversation) ...")
            messages_2b = messages_2a + [
                {"role": "assistant", "content": response_2a},
                {"role": "user", "content": prompt_2b(entry.haiku)},
            ]
            response_2b = multi_turn(model, messages_2b)
            time.sleep(delay)

            # ── Step 4: Cleanup (Prompt 2c) ──────────────────────────────
            print("    → Prompt 2c (cleanup) ...")
            single_turn(model, prompt_2c(entry.haiku))
            time.sleep(delay)

            # ── Step 5: Persist results ──────────────────────────────────
            row = {
                "model": model.name,
                "haiku": entry.haiku,
                "translation": entry.translation,
                "injection": entry.injection,
                "response_1": response_1,
                "response_2b": response_2b,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _append_row(output_csv, row)
            completed += 1
            print(f"    ✓ Row saved ({completed} completed, {skipped} skipped)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")    # ES: this is a nice separator to make the output more readable
    print(f"GENERATION COMPLETE")
    print(f"  Total pairs : {total_pairs}")  # ES: this is the total number of (model, haiku) pairs that were processed
    print(f"  Completed   : {completed}")   # ES: this is the number of (model, haiku) pairs that were successfully processed
    print(f"  Skipped     : {skipped}")     # ES: this is the number of (model, haiku) pairs that were skipped (because they already exist in the output CSV)
    print(f"  Output file : {output_csv}")  # ES: this is the path to the output CSV file where the results were saved
    print(f"{'='*72}\n")    # ES: this is a nice separator to make the output more readable   
