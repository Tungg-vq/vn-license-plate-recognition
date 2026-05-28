import os
import sys
from collections import Counter

TRAIN_FILE = "data/OCR data/rec_train_clean.txt"
VAL_FILE = "data/OCR data/rec_val_clean.txt"
TRAIN_DEDUPE = "data/OCR data/rec_train_final.txt"
VAL_DEDUPE = "data/OCR data/rec_val_final.txt"
KEYS_FILE = "data/OCR data/ppocr_keys.txt"

def read_lines(path):
    if not os.path.exists(path):
        print(f"ERROR: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def parse_line(line):
    parts = line.rsplit(maxsplit=1)
    return tuple(parts) if len(parts) == 2 else None

def main():
    print("=" * 60)
    print("CLEANING DATA & REMOVING OVERLAPS")
    print("=" * 60)
    
    train_lines = read_lines(TRAIN_FILE)
    val_lines = read_lines(VAL_FILE)
    print(f"\nOriginal - Train: {len(train_lines)}, Val: {len(val_lines)}")

    # The training set is our source of truth. We only remove duplicates from the validation set.
    train_set = set(train_lines)
    val_before_count = len(val_lines)

    # Create a new validation list that contains only lines not present in the training set.
    val_clean = [line for line in val_lines if line not in train_set]
    removed_count = val_before_count - len(val_clean)

    print(f"\nData Leakage Found: {removed_count} samples in the validation set also exist in the training set.")
    print(f"These duplicates will be removed from the validation set.")

    print(f"\nCleaned - Train: {len(train_lines)} (unchanged), Val: {len(val_clean)}")

    os.makedirs(os.path.dirname(TRAIN_DEDUPE), exist_ok=True)
    with open(TRAIN_DEDUPE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines)) # Write the original, full training set
    with open(VAL_DEDUPE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_clean))
    print(f"Wrote: {TRAIN_DEDUPE}")
    print(f"Wrote: {VAL_DEDUPE}")
    
    chars = set()
    for line in train_lines + val_clean: # Use original train lines for complete charset
        parsed = parse_line(line)
        if parsed:
            for ch in parsed[1]:
                chars.add(ch)
    
    chars = sorted(chars)
    print(f"\nUnique chars: {len(chars)}")
    print(f"Charset: {''.join(chars)}")
    
    with open(KEYS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chars))
    print(f"Wrote: {KEYS_FILE}")
    
    print("\n" + "=" * 60)
    print("DONE! Update config to use rec_train_final.txt and rec_val_final.txt")
    print("=" * 60)

if __name__ == '__main__':
    main()
