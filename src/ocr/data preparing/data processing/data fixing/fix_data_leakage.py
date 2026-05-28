
import os
import sys


TRAIN_FILE_IN = "data/OCR data/rec_train_clean.txt"
VAL_FILE_IN = "data/OCR data/rec_val_clean.txt"


TRAIN_FILE_OUT = "data/OCR data/rec_train_final.txt"
VAL_FILE_OUT = "data/OCR data/rec_val_final.txt"
KEYS_FILE_OUT = "data/OCR data/ppocr_keys.txt"
# ---

def read_lines(path):
   
    if not os.path.exists(path):
        print(f"✗ ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def main():
    print("\n" + "="*80)
    print(" PADDLEOCR DATA LEAKAGE FIX & CLEANUP SCRIPT")
    print("="*80)


    print(f"\n[1/4] Loading data files...")
    train_lines = read_lines(TRAIN_FILE_IN)
    val_lines = read_lines(VAL_FILE_IN)
    print(f"  ✓ Loaded {len(train_lines):,} samples from training file.")
    print(f"  ✓ Loaded {len(val_lines):,} samples from validation file.")

    
    print("\n[2/4] Removing overlaps from validation set...")
    val_before_count = len(val_lines)
    
    
    overlap = train_lines.intersection(val_lines)
    val_clean = val_lines - overlap
    
    removed_count = val_before_count - len(val_clean)
    
    if removed_count > 0:
        print(f"  ✓ Data leakage found! Removed {removed_count} overlapping samples from validation set.")
    else:
        print("  ✓ No data leakage found between train and validation sets. Good job!")
    
    print(f"  - Original validation size: {val_before_count:,}")
    print(f"  - Final validation size:    {len(val_clean):,}")

  
    print("\n[3/4] Regenerating character dictionary...")
    all_clean_lines = train_lines.union(val_clean)
    chars = set()
    for line in all_clean_lines:
        try:
            label = line.split('\t')[1]
            for char in label:
                chars.add(char)
        except IndexError:
            print(f"  - Warning: Malformed line, skipping for charset generation: {line}")
            continue

    sorted_chars = sorted(list(chars))
    print(f"  ✓ Found {len(sorted_chars)} unique characters.")
    print(f"  - Charset: {''.join(sorted_chars)}")

  
    print("\n[4/4] Writing final, cleaned data files...")
    output_dir = os.path.dirname(TRAIN_FILE_OUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

   
    with open(TRAIN_FILE_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(list(train_lines))))
    print(f"  ✓ Wrote {len(train_lines):,} samples to: {TRAIN_FILE_OUT}")

   
    with open(VAL_FILE_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(list(val_clean))))
    print(f"  ✓ Wrote {len(val_clean):,} samples to: {VAL_FILE_OUT}")


    with open(KEYS_FILE_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_chars))
    print(f"  ✓ Wrote {len(sorted_chars)} characters to: {KEYS_FILE_OUT}")

    print("\n" + "="*80)
    print("  ✓ SUCCESS! Your dataset is now clean and ready for training.")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Update your YAML config file to use the new `_final.txt` files.")
    print("2. Start training. You should now see a normal accuracy curve that starts low and gradually increases.")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()