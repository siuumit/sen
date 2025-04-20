import os
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load models
clf = joblib.load("clf.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['comment']
        data = [message]
        vect = tfidf.transform(data).toarray()
        prediction = clf.predict(vect)
        return render_template('index.html', sentiment=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
