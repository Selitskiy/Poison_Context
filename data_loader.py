"""
Data loader for the Poison Context Haiku experiment.

Reads haiku_combined.csv into a list of HaikuEntry dataclass objects
and validates that every row has non-empty haiku, translation, and injection fields.
"""

import csv
import os
from dataclasses import dataclass


@dataclass
class HaikuEntry:
    haiku: str
    translation: str
    injection: str


# Default path to the CSV relative to this file's directory
_DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "test_haiku_combined.csv")


def load_haiku(csv_path: str = _DEFAULT_CSV) -> list[HaikuEntry]:
    """
    Read haiku_combined.csv and return a list of HaikuEntry objects.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    entries: list[HaikuEntry] = []

    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)

        for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
            haiku = row["haiku"].strip()
            translation = row["translation"].strip()
            injection = row["injection"].strip()

            # Validate: all three fields must be non-empty
            if not haiku:
                raise ValueError(f"Row {row_num}: 'haiku' field is empty")
            if not translation:
                raise ValueError(f"Row {row_num}: 'translation' field is empty")
            if not injection:
                raise ValueError(f"Row {row_num}: 'injection' field is empty")

            entries.append(HaikuEntry(haiku=haiku, translation=translation,
                                      injection=injection))

    return entries

