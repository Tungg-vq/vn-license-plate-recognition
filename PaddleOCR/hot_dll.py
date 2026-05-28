import os
import shutil
import glob

env_path = r"C:\Users\ADMIN\anaconda3\envs\torch310"
# Thêm cái ổ nvidia\cublas\bin vào danh sách đi săn
search_dirs = [
    os.path.join(env_path, r"Library\bin"),
    os.path.join(env_path, r"Lib\site-packages\nvidia\cudnn\bin"),
    os.path.join(env_path, r"Lib\site-packages\nvidia\cublas\bin") 
]

print("🔍 Đang đi săn nốt mấy con DLL lẩn trốn...")
for d in search_dirs:
    if not os.path.exists(d): continue
    # Bế cả lò cuDNN, cuBLAS và zlib
    for pattern in ["cudnn*.dll", "cublas*.dll", "zlibwapi.dll"]:
        for f in glob.glob(os.path.join(d, pattern)):
            shutil.copy(f, ".")
            print(f"✅ Đã bế: {os.path.basename(f)}")
print("\n🔥 Sạch bách! Lên lửa đi sếp!")