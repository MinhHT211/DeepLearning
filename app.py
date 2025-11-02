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

    with open(HistoryFile, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    word = ""  # Used to save input from user

    if request.method == "POST":
        word = request.form.get("text", "").strip()

        if word:
            seq = text_to_sequence(word)

            if len(seq.shape) == 2:
                seq = np.expand_dims(seq, axis=0)
            
            # preds: shape (n_lang,)
            preds = model.predict(seq, verbose=0)[0]

            # preds = model.predict(seq)
            lang_index = np.argmax(preds)
            confidence = float(np.max(preds)) * 100
            prediction = LANGUAGES[lang_index]
            all_confidences = {LANGUAGES[i]: round(float(preds[i]) * 100, 2) for i in range(len(LANGUAGES))}

            record = {
                "text": word,
                "language": prediction,
                "confidence": round(confidence, 2),
                "all_confidences": all_confidences,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_history(record)

    # Retrieve word to HTML
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        word=word,  # Pass the word back to HTML
        all_confidences=all_confidences if word else None
    )


@app.route("/history")
def show_history():
    if not os.path.exists(HistoryFile):
        return "No history yet."
    with open(HistoryFile, encoding="utf-8") as f:
        history = json.load(f)

    history = list(reversed(history[-200:]))
    return render_template("history.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)
