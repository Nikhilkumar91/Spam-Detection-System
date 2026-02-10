from flask import Flask, render_template, request, jsonify
import string
import numpy as np
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once if not already present
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# -------------------------------
# Load model at startup
# -------------------------------
MODEL_PATH = "C:\\Users\\nikhi\\Downloads\\Data science practice\\Spam Detection\\spam_model.h5"
model = load_model(MODEL_PATH)

# Must be SAME as training
DIC_SIZE = 2000
MAX_LEN = 50   # use your training max_len

lem = WordNetLemmatizer()

# -------------------------------
# Text preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join(
        lem.lemmatize(word)
        for word in text.split()
        if word not in stopwords.words('english')
    )
    return text

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mail = data.get("message", "").strip()

    if not mail:
        return jsonify({
            "error": "Please enter an email or message to check."
        }), 400

    # Preprocess
    text = clean_text(mail)
    v = [one_hot(text, DIC_SIZE)]
    p = pad_sequences(v, maxlen=MAX_LEN, padding='post')

    # Predict
    probability = model.predict(p)[0][0]

    if probability >= 0.5:
        result = "Ham(Good Mail)"
    else:
        result = "Spam(Bad Mail)"

    return jsonify({
        "result": result,
        "probability": round(float(probability) * 100, 2)
    })


if __name__ == '__main__':
    app.run(debug=True)
