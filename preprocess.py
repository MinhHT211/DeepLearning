import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

LANGUAGES = [
    "Arabic",
    "Chinese",
    "English",
    "French",
    "German",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
    "Vietnam",
    "Norwegian",
    "Luxembourg",
    "Irish",
    "Swedish",
    "Czech",
    "Romanian",
    "Slovak",
    
]

MAX_LEN = 15
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệ"
    "ìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữựỳýỷỹỵ"
    "çñüß"
    "абвгдезийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"  
    "一二三四五六七八九十的我你是了不在这他她它"  
    "あいうえおかきくけこさしすせそたちつてと"  
    "アイウエオカキクケコサシスセソタチツテト"  
    "가나다라마바사아자차카타파하"  
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئةى"
    "Æ Ø Å"
    "ä é è ë ï ö ü ß ç"
    "á é í ó ú bh ch dh fh gh mh ph sh th"
    "Å å Ä ä Ö ö é à ü è ô"
    "á é í ó ú ý ů č ď ě ň ř š ť ž"
    "ă â î ș ț Ă Â Î Ș Ț"
    "á é í ó ú ý ĺ ŕ ä ô č ď ľ ň š ť ž ch Á É Í Ó Ú Ý Ĺ Ŕ Ä Ô Č Ď Ľ Ň Š Ť Ž CH"
)
CHAR_DICT = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(LANGUAGES)

def encode_word(word):
    word = word.lower()[:MAX_LEN]
    x = np.zeros((MAX_LEN, len(CHARS)))
    for i, ch in enumerate(word):
        if ch in CHAR_DICT:
            x[i, CHAR_DICT[ch]] = 1
    return x

def load_data(data_folder="data"):
    X, y = [], []
    for idx, lang in enumerate(LANGUAGES):
        path = os.path.join(data_folder, f"{lang}.txt")
        with open(path, encoding="utf-8") as f:
            words = [w.strip() for w in f.readlines() if w.strip()]
        for w in words:
            X.append(encode_word(w))
            y.append(idx)
    X = np.array(X)
    y = to_categorical(y, NUM_CLASSES)
    return train_test_split(X, y, test_size=0.15, random_state=42)

def text_to_sequence(text):
    # Function to convert an input sequence into a one-hot matrix (like when training)
    text = text.lower()[:MAX_LEN]
    x = np.zeros((MAX_LEN, len(CHARS)))
    for i, ch in enumerate(text):
        if ch in CHAR_DICT:
            x[i, CHAR_DICT[ch]] = 1
    return np.expand_dims(x, axis=0)  # to match model.predict()
