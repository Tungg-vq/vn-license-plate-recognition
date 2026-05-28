import os
import sys



train_txt = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'OCR data', 'rec_train_clean.txt')
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join('data', 'OCR data', 'ppocr_keys.txt')

if not os.path.exists(train_txt):
    print(f"Train file not found: {train_txt}")
    sys.exit(1)

chars = set()
with open(train_txt, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        label = parts[1]
        for ch in label:
            chars.add(ch)

if len(chars) == 0:
    print('No characters found in training file')
    sys.exit(1)

sorted_chars = sorted(chars)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    for ch in sorted_chars:
        f.write(ch + '\n')

print(f'Wrote {len(sorted_chars)} characters to {out_path}')
