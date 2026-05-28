#!/usr/bin/env python3
import os
import sys
from collections import Counter

TRAIN_FILE = "data/OCR data/rec_train_clean.txt"
VAL_FILE = "data/OCR data/rec_val_clean.txt"
TRAIN_DEDUPE = "data/OCR data/rec_train_final.txt"
VAL_DEDUPE = "data/OCR data/rec_val_final.txt"
KEYS_FILE = "data/OCR data/ppocr_keys.txt"

def read_lines(path):
    """Read lines from file, strip whitespace."""
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def parse_line(line):
    """Split into (img_path, label). Return None if malformed."""
    parts = line.rsplit(maxsplit=1)
    if len(parts) != 2:
        return None
    return tuple(parts)

def validate_and_clean(train_lines, val_lines):
    
    report = {
        'train_orig': len(train_lines),
        'val_orig': len(val_lines),
        'train_malformed': 0,
        'val_malformed': 0,
        'train_missing_files': 0,
        'val_missing_files': 0,
        'overlap_count': 0,
        'train_final': 0,
        'val_final': 0,
    }
    
   
    train_parsed = []
    for line in train_lines:
        parsed = parse_line(line)
        if parsed is None:
            report['train_malformed'] += 1
            continue
        img_path, label = parsed
        if not os.path.exists(img_path):
            report['train_missing_files'] += 1
            continue
        train_parsed.append(line)
    

    val_parsed = []
    for line in val_lines:
        parsed = parse_line(line)
        if parsed is None:
            report['val_malformed'] += 1
            continue
        img_path, label = parsed
        if not os.path.exists(img_path):
            report['val_missing_files'] += 1
            continue
        val_parsed.append(line)
    
    train_set = set(train_parsed)
    val_set = set(val_parsed)
    overlap_exact = train_set & val_set
    report['overlap_count'] = len(overlap_exact)
    
    train_imgs = {line.rsplit(maxsplit=1)[0]: line for line in train_parsed}
    val_imgs = {line.rsplit(maxsplit=1)[0]: line for line in val_parsed}
    overlap_files = set(train_imgs.keys()) & set(val_imgs.keys())
    
    train_clean = [l for l in train_parsed if l not in overlap_exact]
    val_clean = [l for l in val_parsed 
                 if l not in overlap_exact and l.rsplit(maxsplit=1)[0] not in overlap_files]
    
    report['train_final'] = len(train_clean)
    report['val_final'] = len(val_clean)
    
    return train_clean, val_clean, report

def extract_chars(lines):
    chars = set()
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            _, label = parsed
            for ch in label:
                chars.add(ch)
    return sorted(chars)

def main():
    print("=" * 60)
    print("PADDLEOCR DATA CLEANING & VALIDATION")
    print("=" * 60)
    
    # Read files
    print("\n[1/5] Reading data files...")
    train_lines = read_lines(TRAIN_FILE)
    val_lines = read_lines(VAL_FILE)
    print(f"  Train: {len(train_lines)} lines")
    print(f"  Val: {len(val_lines)} lines")
    
    # Validate and clean
    print("\n[2/5] Validating and removing overlaps...")
    train_clean, val_clean, report = validate_and_clean(train_lines, val_lines)
    print(f"  Train malformed: {report['train_malformed']}")
    print(f"  Train missing files: {report['train_missing_files']}")
    print(f"  Val malformed: {report['val_malformed']}")
    print(f"  Val missing files: {report['val_missing_files']}")
    print(f"  Exact line overlaps removed: {report['overlap_count']}")
    print(f"  Train final: {report['train_final']} (was {report['train_orig']})")
    print(f"  Val final: {report['val_final']} (was {report['val_orig']})")
    
    # Write cleaned files
    print("\n[3/5] Writing cleaned files...")
    os.makedirs(os.path.dirname(TRAIN_DEDUPE), exist_ok=True)
    with open(TRAIN_DEDUPE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_clean))
    print(f"  Wrote: {TRAIN_DEDUPE}")
    
    with open(VAL_DEDUPE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_clean))
    print(f"  Wrote: {VAL_DEDUPE}")
    
    # Extract characters
    print("\n[4/5] Extracting character set...")
    all_chars = extract_chars(train_clean + val_clean)
    print(f"  Unique characters: {len(all_chars)}")
    print(f"  Charset: {''.join(all_chars)}")
    
    # Write charset
    print("\n[5/5] Writing character dictionary...")
    with open(KEYS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_chars))
    print(f"  Wrote: {KEYS_FILE}")
    
    # Label distribution
    print("\n" + "=" * 60)
    print("LABEL STATISTICS")
    print("=" * 60)
    train_labels = [parse_line(l)[1] for l in train_clean if parse_line(l)]
    val_labels = [parse_line(l)[1] for l in val_clean if parse_line(l)]
    
    print(f"\nTrain label lengths:")
    lengths = Counter([len(l) for l in train_labels])
    for length in sorted(lengths.keys()):
        print(f"  Length {length}: {lengths[length]} samples")
    
    print(f"\nVal label lengths:")
    lengths = Counter([len(l) for l in val_labels])
    for length in sorted(lengths.keys()):
        print(f"  Length {length}: {lengths[length]} samples")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Update config to use:")
    print(f"   - Train: data/OCR data/rec_train_final.txt")
    print(f"   - Val: data/OCR data/rec_val_final.txt")
    print(f"   - Chars: {KEYS_FILE}")
    print(f"2. Run training:")
    print(f"   python PaddleOCR\\tools\\train.py -c PaddleOCR\\configs\\rec\\finetune_user_rec.yml")
    print("=" * 60)

if __name__ == '__main__':
    main()
