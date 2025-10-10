import os

import re

import csv
# Thư mục chứa các file .txt
folder = "data"

print("📊 Số lượng từ trong từng file:\n")

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tách từ bằng khoảng trắng
        words = text.split()
        word_count = len(words)

        print(f"{filename}: {word_count} từ")
import re
import os
import string

# 📂 Thư mục chứa các file cần xử lý
folder_path = r"C:\Users\ASUS\OneDrive\Documents\DeepLearning\data"

# 🔤 Tạo regex xóa dấu câu an toàn (giữ nguyên ký tự chữ Unicode)
# Bao gồm: . , ? ! ( ) [ ] { } " ' ; : … và các biến thể
punctuation_pattern = re.compile(r"[{}“”‘’、。？！：；（）［］【】✓ ﻿ ➟《》,‥…‧]".format(re.escape(string.punctuation)))

# 🔁 Duyệt qua tất cả file .txt trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        print(f"🧹 Đang xử lý: {filename}")

        # Đọc nội dung gốc
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Xóa các dấu câu, giữ nguyên ký tự ngôn ngữ
        clean_text = re.sub(punctuation_pattern, "", text)

        # Ghi đè lại file gốc
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

print("✅ Đã xóa dấu câu (. , ? ! () …) mà không ảnh hưởng tới ngôn ngữ trong:", folder_path)
