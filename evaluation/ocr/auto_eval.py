import os
from paddlex import create_model

# --- Configuration ---
# Load the trained model.
# Ensure this is the correct path to your exported inference model.
model = create_model(
    model_name='en_PP-OCRv4_mobile_rec',
    model_dir='inference/model_chot_ha'
)

# Path to the validation ground truth file.
# This file should contain lines in the format: "path/to/image.jpg\tlabel"
GT_FILE = "data/OCR data/rec_val_final.txt"

# --- Main Evaluation Logic ---
if __name__ == '__main__':
    # 1. Load Ground Truth labels into a dictionary
    # The key is the full image path, and the value is the label.
    gt_data = {}
    if not os.path.exists(GT_FILE):
        print(f" ERROR: Ground truth file not found at: {GT_FILE}")
        exit()

    with open(GT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_path, label = parts
                # Normalize path separators for consistency
                gt_data[os.path.normpath(img_path)] = label

    if not gt_data:
        print(" ERROR: No data loaded from the ground truth file. Is it empty or in the wrong format?")
        exit()

    print(f" Loaded {len(gt_data)} labels from '{os.path.basename(GT_FILE)}'.")
    print(" Starting evaluation on the validation set...")
    print("-" * 50)

    # 2. Iterate through labels, predict, and compare
    correct_predictions = 0
    total_samples = 0

    for img_path, true_label in gt_data.items():
        # Check if the image specified in the label file actually exists
        if not os.path.exists(img_path):
            print(f" WARNING: Image not found, skipping: {img_path}")
            continue

        total_samples += 1

        try:
            # Predict the text from the image
            results = model.predict(img_path)

            # The model returns a list of dictionaries. We concatenate the text.
            pred_text = "".join([res['rec_text'] for res in results])

            # 3. Compare prediction with ground truth
            if pred_text == true_label:
                correct_predictions += 1
            else:
                # Print mismatches for analysis
                img_filename = os.path.basename(img_path)
                print(f" MISMATCH: {img_filename}")
                print(f"   - Ground Truth: {true_label}")
                print(f"   - Predicted:    {pred_text}")
                print("-" * 20)

        except Exception as e:
            print(f" ERROR processing {os.path.basename(img_path)}: {e}")

    # 4. Calculate and report final accuracy
    print("=" * 50)
    print(" EVALUATION SUMMARY")
    print("=" * 50)

    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        print(f"   - Total Images Evaluated: {total_samples}")
        print(f"   - Correct Predictions:    {correct_predictions}")
        print(f"   - Incorrect Predictions:  {total_samples - correct_predictions}")
        print(f"   -  Validation Accuracy: {accuracy:.2f}%")
    else:
        print("No valid images were found to evaluate. Please check your paths and the ground truth file.")