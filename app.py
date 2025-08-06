from flask import Flask, render_template, request
import joblib
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

@app.route('/')
def home():
    return render_template('index.html', prediction=None, news='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        vectorized_news = vectorizer.transform([news])
        prediction = model.predict(vectorized_news)
        result = "Fake News ❌" if prediction[0] == 1 else "Real News ✅"
        return render_template('index.html', prediction=result, news=news)

@app.route('/realtime')
def realtime_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'pageSize': 5,  
    }
    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])
    
    predictions = []
    for article in articles:
        title = article.get('title', '')
        desc = article.get('description', '')
        content = f"{title} {desc}"
        if content.strip():
            vectorized = vectorizer.transform([content])
            pred = model.predict(vectorized)
            label = "Fake News ❌" if pred[0] == 1 else "Real News ✅"
            predictions.append({'title': title, 'desc': desc, 'label': label})
    
    return render_template('realtime.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)