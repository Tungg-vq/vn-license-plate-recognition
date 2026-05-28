FILE_CU = "data/OCR data/rec_train_final.txt"
FILE_MOI = "data/synth_line1_label.txt"
FILE_CHOT = "data/train_sieu_cap.txt"

with open(FILE_CHOT, 'w', encoding='utf-8', newline='\n') as f_out:
    # Bơm data cũ vào
    with open(FILE_CU, 'r', encoding='utf-8') as f1:
        f_out.write(f1.read())
    # Bơm data mới vào
    with open(FILE_MOI, 'r', encoding='utf-8') as f2:
        if not f_out.tell() == 0: f_out.write('\n') # Tránh dính dòng
        f_out.write(f2.read())

print(f"Done: {FILE_CHOT}")