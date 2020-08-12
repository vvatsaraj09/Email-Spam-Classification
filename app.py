import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.form["body"]
    prediction = model.predict(cv.transform([body]))
    return render_template("index.html", status=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
