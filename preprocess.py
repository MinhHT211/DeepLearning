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
    "Swahil",
    "Ukraina",
    "Persian",
    "Hebrew",
    "Finnish",
    "Bengali",
    "Polish"
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
    "A Ą B C Ć D E Ę F G H I J K L Ł M N Ń O Ó P R S Ś T U W Y Z Ź Żswah"
    "б,в,г,ґ,д,ж,з,й,к,л,м,н,п,р,с,т,ф,х,ц,ч,ш,щ,а,е,є,и,і,ї,о,у,ю,я"
    "B, D, F, G, H, J, K, L, M, N, P, R, S, T, V, W, Y, Z, A, I, E, O, U, Ch, Dh, Gh, Ng', Ny, Sh, Th"
    "ا‎ ‎ب‎ ‎پ‎ ‎ت‎ ‎ث‎ ‎ج‎ ‎چ‎ ‎ح‎ ‎خ‎ ‎د‎ ‎ذ‎ ‎ر‎ ‎ز‎ ‎ژ‎ ‎س‎ ‎ش‎ ‎ص‎ ‎ض‎ ‎ط‎ ‎ظ‎ ‎ع‎ ‎غ‎ ‎ف‎ ‎ق‎ ‎ک‎ ‎گ‎ ‎ل‎ ‎م‎ ‎ن‎ ‎و‎ ‎ه‎ ‎ی"
    "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת"
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Å Ä Ö"
    "অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ ড় ঢ় য় ৎ ং ঃ ঁ ক খ গ ঘ ঙচ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল শ ষ স হ ড় ঢ় য় "

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
    X = np.array(X, dtype=np.float32)
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
