import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

LANGUAGES = [
    "arabic",
    "chinese",
    "russian",
    "english",
    "french",
    "german",
    "japanese",
    "russian",
    "spanish",
    "vietnam",
    "armenian",
    "uzbek",
    "sinhala",
    "malagasy",
    "azerbaijani",
    "amharic",
    "kurdish"
]

MAX_LEN = 15

CHARS = (
    # --- Latin cơ bản (dùng cho Uzbek, Malagasy, Kurdish, Azerbaijani) ---
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # --- Ký tự Latin mở rộng ---
    "çğıİöşüəʻʼ’êîû"
    
    # --- Armenian (Հայերեն) ---
    "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔևՕՖ"
    "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքևօֆ"
    
    # --- Sinhala (සිංහල) ---
    "අආඇඈඉඊඋඌඑඒඔඕඖඓඛගඝඞඟචඡජඣඤඥටඨඩඪණතථදධනපඵබභමයරලවශෂසහළෆ"
    "්ාැෑිීුූෙේොෝෞෟ"
    
    # --- Amharic (አማርኛ) ---
    "ሀሁሂሃሄህሆለሉሊላሌልሎመሙሚማሜምሞ"
    "ሠሡሢሣሤሥሦረሩሪራሬርሮ"
    "ሰሱሲሳሴስሶሸሹሺሻሼሽሾ"
    "ቀቁቂቃቄቅቆበቡቢባቤብቦ"
    "ነኑኒናኔንኖአኡኢኣኤእኦ"
    "ከኩኪካኬክኮወዉዊዋዌውዎ"
    "ዐዑዒዓዔዕዖዘዙዚዛዜዝዞ"
    "የዩዪያዬይዮ"
    
    # --- Ký tự đặc biệt, dấu, punctuation ---
    "՛՝՞՜։«»–…፡።፣፤፥፦፧ "
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
