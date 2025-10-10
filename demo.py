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

file_path = r"C:\Users\ASUS\Downloads\vietnam.txt"

# Đọc toàn bộ file
with open(file_path, "r", encoding="utf-8") as tsv:
    reader = csv.reader(tsv, delimiter="\t")
    lines = []
    for row in reader:
        if len(row) >= 3:
            lines.append(row[2].strip())

# Ghi đè lại chính file đó (chỉ còn câu tiếng Việt)
with open(file_path, "w", encoding="utf-8") as txt:
    for line in lines:
        txt.write(line + "\n")

print("✅ Đã xóa ID, mã 'vie' và các cột khác. File gốc đã được cập nhật:", file_path)
