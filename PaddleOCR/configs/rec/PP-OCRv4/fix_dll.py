import os
import shutil

# Tự động mò vào đúng môi trường torch310 của mày
env_path = os.environ.get('CONDA_PREFIX', r'C:\Users\ADMIN\anaconda3\envs\torch310')
dll_name = "cudnn64_8.dll"

# Những cái ngách mà Windows thường cất giấu file này
sources = [
    os.path.join(env_path, "Library", "bin", dll_name),
    os.path.join(env_path, "Lib", "site-packages", "torch", "lib", dll_name),
    os.path.join(env_path, "Lib", "site-packages", "nvidia", "cudnn", "bin", dll_name)
]

dest = os.path.join(os.getcwd(), dll_name)
found = False

for src in sources:
    if os.path.exists(src):
        shutil.copy(src, dest)
        print("-" * 50)
        print(f"✅ ĐÃ TÓM ĐƯỢC FILE TẠI:\n-> {src}")
        print(f"✅ Đã ném {dll_name} ra thư mục gốc PaddleOCR!")
        print("🔥 Bấm lệnh Train lại được rồi sếp!")
        print("-" * 50)
        found = True
        break

if not found:
    print("❌ Quét nát máy vẫn không thấy cudnn64_8.dll.")
    print("👉 Do conda ban nãy tải nhầm bản số 9. Mày gõ lệnh này để tải đúng bản số 8 nhé:")
    print("   pip install nvidia-cudnn-cu11==8.9.2.26")
    print("👉 Chạy xong lệnh pip thì chạy lại file fix_dll.py này 1 lần nữa là ăn 100%.")