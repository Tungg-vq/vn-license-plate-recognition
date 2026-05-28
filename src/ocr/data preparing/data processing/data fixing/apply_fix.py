#!/usr/bin/env python3
import sys
import os

print("\n" + "="*70)
print("STEP 1: Reading data files...")
print("="*70)

# Read both files
try:
    with open('data/OCR data/rec_train_clean.txt', 'r', encoding='utf-8') as f:
        train_lines = set(f.read().strip().split('\n'))
    print(f"✓ Train loaded: {len(train_lines)} samples")
except Exception as e:
    print(f"✗ Error reading train: {e}")
    sys.exit(1)

try:
    with open('data/OCR data/rec_val_clean.txt', 'r', encoding='utf-8') as f:
        val_lines = [l for l in f.read().strip().split('\n') if l]
    print(f"✓ Val loaded: {len(val_lines)} samples")
except Exception as e:
    print(f"✗ Error reading val: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 2: Removing data leakage (overlaps)...")
print("="*70)

# Find overlaps
val_before = len(val_lines)
val_clean = [l for l in val_lines if l not in train_lines]
overlaps = val_before - len(val_clean)

print(f"Val before dedup: {val_before} samples")
print(f"Val after dedup:  {len(val_clean)} samples")
print(f"Overlaps removed: {overlaps}")

print("\n" + "="*70)
print("STEP 3: Extracting characters...")
print("="*70)

# Extract unique chars
chars = set()
for line in list(train_lines) + val_clean:
    try:
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            label = parts[1]
            for ch in label:
                chars.add(ch)
    except:
        pass

chars = sorted(chars)
print(f"Unique characters: {len(chars)}")
print(f"Charset: {''.join(chars[:50])}..." if len(chars) > 50 else f"Charset: {''.join(chars)}")

print("\n" + "="*70)
print("STEP 4: Writing output files...")
print("="*70)


os.makedirs('data/OCR data', exist_ok=True)

try:
    with open('data/OCR data/rec_train_final.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(train_lines)))
    print("✓ Written: data/OCR data/rec_train_final.txt")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    with open('data/OCR data/rec_val_final.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_clean))
    print("✓ Written: data/OCR data/rec_val_final.txt")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    with open('data/OCR data/ppocr_keys.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(chars))
    print(f"✓ Written: data/OCR data/ppocr_keys.txt ({len(chars)} chars)")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print(f"\nData Summary:")
print(f"  Train: {len(train_lines)} samples")
print(f"  Val:   {len(val_clean)} samples (cleaned)")
print(f"  Chars: {len(chars)} unique characters")
print(f"\nFixed Issues:")
print(f"  ✓ Removed {overlaps} duplicate samples from validation")
print(f"  ✓ Regenerated character dictionary")
print(f"  ✓ Config already updated with new hyperparameters")
print(f"\nNext: Run training command")
print("="*70 + "\n")