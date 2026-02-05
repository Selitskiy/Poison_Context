"""
Haiku Combination Script

Parses haiku_translation.txt, filters to fully-translated entries,
shuffles with a fixed seed for reproducibility, and outputs to CSV.
"""

import csv
import random
from pathlib import Path

# Fixed seed
RANDOM_SEED = 100

def parse_haiku_file(filepath: Path) -> list[dict]:
    """
    Parse haiku_translation.txt and return entries with all 3 components.
    
    Each line should have format: haiku|translation|injection
    """
    entries = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            
            # Only include entries with exactly 3 non-empty components
            if len(parts) == 3 and all(part.strip() for part in parts):
                entries.append({
                    'haiku': parts[0].strip(),
                    'translation': parts[1].strip(),
                    'injection': parts[2].strip()
                })
    
    return entries


def shuffle_entries(entries: list[dict], seed: int) -> list[dict]:
    """Shuffle entries with a fixed seed for reproducibility."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)
    return shuffled


def write_csv(entries: list[dict], filepath: Path) -> None:
    """Write entries to CSV file."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['haiku', 'translation', 'injection'])
        writer.writeheader()
        writer.writerows(entries)


def main():
    # Resolve paths relative to this script's location
    script_dir = Path(__file__).parent  # parent (\research) of the script directory (\research\combine_haiku.py)
    data_dir = script_dir / 'data'
    input_file = data_dir / 'haiku_translation.txt'
    output_file = data_dir / 'haiku_combined.csv'
    
    # Parse input file
    print(f"Reading from: {input_file}")
    entries = parse_haiku_file(input_file)
    print(f"Found {len(entries)} complete entries (haiku + translation + injection)")
    
    # Shuffle with fixed seed
    shuffled_entries = shuffle_entries(entries, RANDOM_SEED)
    print(f"Shuffled entries with seed: {RANDOM_SEED}")
    
    # Write output
    write_csv(shuffled_entries, output_file)
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    main()

