from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load vectorizer and model
with open("TfidfVectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("web.html")  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email content provided"}), 400

    input_data = vectorizer.transform([email_text])
    prediction = model.predict(input_data)[0]

    result = "Ham" if prediction == 1 else "Spam"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
