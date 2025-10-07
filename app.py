from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from preprocess import text_to_sequence, LANGUAGES
import json, os, datetime
app = Flask(__name__)

# Load trained model
model = load_model("cnn_language_model.h5")

HistoryFile = "artifacts/history.json"
def save_history(record):
    os.makedirs("artifacts", exist_ok=True)
    try:
        with open(HistoryFile, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []

    history.append(record)
    if len(history) > 1000:  
        history = history[-1000:]

# change data Python to JSON anh write to file
    with open(HistoryFile, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]
        seq = text_to_sequence(text)

        #If seq has shape (15, 26) then add batch dimension
        if len(seq.shape) == 2:
            seq = np.expand_dims(seq, axis=0)

        preds = model.predict(seq)

        lang_index = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        prediction = LANGUAGES[lang_index]

        # Save to history
        record = {
            "text": text,
            "language": prediction,
            "confidence": round(confidence, 2),
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_history(record)

    return render_template("index.html", prediction=prediction, confidence=confidence)

@app.route("/history")
def show_history():
    if not os.path.exists(HistoryFile):
        return "No history yet."
    with open(HistoryFile, encoding="utf-8") as f:
        history = json.load(f)

    history = list(reversed(history[-100:]))
    return render_template("history.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)
