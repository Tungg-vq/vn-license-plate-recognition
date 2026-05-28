import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_DIR = "data/YOLO data/labels/train"
VAL_DIR = "data/YOLO data/labels/val"

def load_labels(label_dir, split_name):
    data = []
    if not os.path.exists(label_dir):
        return pd.DataFrame()
        
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    data.append({
                        "Dataset": split_name, 
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height
                    })
    return pd.DataFrame(data)

df_train = load_labels(TRAIN_DIR, "Train")
df_val = load_labels(VAL_DIR, "Validation")

df_combined = pd.concat([df_train, df_val], ignore_index=True)

if df_combined.empty:
    print("no label found")
else:
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    
    sns.scatterplot(
        data=df_combined, x="x_center", y="y_center", 
        hue="Dataset", palette={"Train": "royalblue", "Validation": "darkorange"},
        alpha=0.6, edgecolor=None, ax=axes[0]
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1, 0) 
    axes[0].set_title("License plate position comparision (Train vs Val)", fontweight='bold')

    
    sns.scatterplot(
        data=df_combined, x="width", y="height", 
        hue="Dataset", palette={"Train": "royalblue", "Validation": "darkorange"},
        alpha=0.6, edgecolor=None, ax=axes[1]
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("License plate size comparision (Train vs Val)", fontweight='bold')

    plt.tight_layout()
    plt.savefig("train_val_comparison.png", dpi=300)
    plt.show()