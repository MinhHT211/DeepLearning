
import pandas as pd
import os

csv_path = os.path.expanduser("C:/Users/ASUS/Downloads/archive/dataset.csv")

os.makedirs("data", exist_ok=True)

df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip()

text_col = "Text"
lang_col = "language"

df = df.dropna(subset=[text_col, lang_col])
df = df[df[text_col].str.strip() != ""]


print(f"📊 Total: {len(df)}")
print(f"🌍 Languages: {df[lang_col].nunique()}")

for lang in sorted(df[lang_col].unique()):
    subset = df[df[lang_col] == lang]
    filename = f"data/{lang.lower().replace(' ', '_')}.txt"

    subset[text_col].to_csv(filename, index=False, header=False, encoding="utf-8")

    print(f"✅ {lang}: {len(subset)} lines → {filename}")

print("\n🎉 Completed")
