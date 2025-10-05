from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from preprocess import text_to_sequence, LANGUAGES

app = Flask(__name__)

# Load trained model
model = load_model("cnn_language_model.h5")

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

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
