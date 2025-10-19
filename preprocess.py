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
    "Italian",
    "Hindi",
    "Thai",
    "Dutch",
    "Turkish",
    "Greek",
    "Indonesian"
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
    "à, è, é, ì, ò, ù"
    "ा, ि, ी, ु, ू, ृ, े, ै, ो, ौ, ं, ः, ँ"
    "ก,ข,ฃ,ค,ฅ,ฆ,ง,จ,ฉ,ช,ซ,ฌ,ญ,ฎ,ฏ,ฐ,ฑ,ฒ,ณ,ด,ต,ถ,ท,ธ,น,บ,ป,ผ,ฝ,พ,ฟ,ภ,ม,ย,ร,ล,ว,ศ,ษ,ส,ห,ฬ,อ,ฮ,ะ,า,ิ,ี,ึ,ื,ุ,ู,เ,แ,โ,ใ,ไ,ำ,เา,เอะ,เอ,แอะ,แอ,โอะ,โอ,เาะ,ออ,เอาะ,เออ,เอียะ,เอีย,เอือะ,เอือ,อัวะ,อัว,่,้,๊,๋,ๆ,ฯ,ฯลฯ,๏,๚,๛"
    "ë, ï, ö, ü, é, á, ó, ú, è"
    "ç, Ç, ğ, Ğ, ı, İ, ö, Ö, ş, Ş, ü, Ü"
    "ά, έ, ή, ί, ό, ύ, ώ, ϊ, ϋ, Ά, Έ, Ή, Ί, Ό, Ύ, Ώ, Ϊ, Ϋ"
    "é"
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
