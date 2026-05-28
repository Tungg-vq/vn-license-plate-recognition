import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

CROP_CSV= 'crop_labels.csv'
GEN_CSV= 'gen_labels.csv'
CROP_DIR= 'cropped'
GEN_DIR= 'generated'
PATH= 'data/OCR data'

OUTPUT_DIR= 'data/OCR data'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
def process_csv(csv_file, folder_name):
    df= pd.read_csv(f"{PATH}/{csv_file}", dtype= str)
    formatted_data= []
    for index, row in df.iterrows():
        fname= str(row[0]).strip()
        label= str(row[1])
        clean_label= re.sub(r'[^A-Z0-9]', '', label.upper())
        full_path= f"{PATH}/{folder_name}/{fname}"
        formatted_data.append(f"{full_path}\t{clean_label}")
    return formatted_data

try:
    crop_data= process_csv(CROP_CSV, CROP_DIR)
    gen_data= process_csv(GEN_CSV, GEN_DIR)
    all_data= crop_data+ gen_data
    print(f"Total samples: {len(all_data)}")

    train_data, val_data= train_test_split(all_data, test_size= 0.2, random_state= 42 )
    with open(f"{OUTPUT_DIR}/rec_train.txt", 'w', encoding= 'utf-8') as f:
        f.write('\n'.join(train_data))
    with open(f"{OUTPUT_DIR}/rec_val.txt", 'w', encoding= 'utf-8') as f:
        f.write('\n'.join(val_data))
    print("Data preparation completed successfully.")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
except Exception as e:
    print(f"Error during data preparation: {e}")

