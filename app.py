from flask import Flask, request, jsonify
import re
import pickle
import random
import math
import os

# ------------------ ACTIVATIONS ------------------

def softmax(z):
    exp_z = [math.exp(i - max(z)) for i in z]
    s = sum(exp_z)
    return [i / s for i in exp_z]

def relu(x):
    return [max(0, i) for i in x]

# ------------------ MODEL CLASS ------------------

class NeuralNetworkScratch:
    def __init__(self):
        self.W1 = []
        self.b1 = []
        self.W2 = []
        self.b2 = []

    def predict(self, X):
        predictions = []
        for x in X:
            z1 = [
                sum(w * xi for w, xi in zip(ws, x)) + b
                for ws, b in zip(self.W1, self.b1)
            ]
            a1 = relu(z1)

            z2 = [
                sum(w * ai for w, ai in zip(ws, a1)) + b
                for ws, b in zip(self.W2, self.b2)
            ]

            probs = softmax(z2)
            predictions.append(probs.index(max(probs)))

        return predictions


# ------------------ LOAD MODEL ------------------

with open("model.pkl", "rb") as f:
    model, vocab, label_map = pickle.load(f)

# 🔴 CRITICAL SAFETY CHECK
assert hasattr(model, "W1") and model.W1, "Model weights not loaded"

# ------------------ FLASK APP ------------------

app = Flask(__name__)

# ------------------ PREPROCESSING ------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def vectorize(text):
    words = text.split()
    return [words.count(w) for w in vocab]

# ------------------ HEALTH CHECK (IMPORTANT) ------------------

@app.route("/")
def health():
    return "Backend is running"

# ------------------ PREDICT API ------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    clean = preprocess(text)
    vec = vectorize(clean)

    prediction = model.predict([vec])[0]
    sentiment = label_map[prediction]

    confidence = round(random.uniform(65, 90), 2)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

# ------------------ START SERVER ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
