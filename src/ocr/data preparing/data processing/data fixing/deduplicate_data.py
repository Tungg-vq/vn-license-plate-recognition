import os
import sys

# Remove val samples that appear in train (data leakage fix)
train_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'OCR data', 'rec_train_clean.txt')
val_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'OCR data', 'rec_val_clean.txt')
val_out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join('data', 'OCR data', 'rec_val_clean_dedup.txt')

if not os.path.exists(train_path):
    print(f"Train file not found: {train_path}")
    sys.exit(1)
if not os.path.exists(val_path):
    print(f"Val file not found: {val_path}")
    sys.exit(1)

# Read train lines as a set
train_lines = set()
with open(train_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            train_lines.add(line)

print(f"Loaded {len(train_lines)} train samples")

# Read val lines and keep only those NOT in train
val_lines = []
removed = 0
with open(val_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and line not in train_lines:
            val_lines.append(line)
        elif line:
            removed += 1

print(f"Loaded {len(val_lines) + removed} val samples, removed {removed} duplicates")
print(f"Writing {len(val_lines)} deduplicated val samples to {val_out_path}")

# Write deduplicated val
os.makedirs(os.path.dirname(val_out_path), exist_ok=True)
with open(val_out_path, 'w', encoding='utf-8') as f:
    for line in val_lines:
        f.write(line + '\n')

print(f"Done. Backup original with: ren {val_path} {val_path}.bak")
print(f"Then replace with: ren {val_out_path} {val_path}")
