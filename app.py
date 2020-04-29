from flask import Flask, render_template, request, url_for
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/make_predictions', methods=['POST'])
def make_predictions():
    df = pd.read_csv('./dataset/spam.csv', encoding='latin-1').drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

    X = df.message
    y = LabelEncoder().fit_transform(df['class'])

    model = Pipeline([
        ('cv', CountVectorizer(stop_words=STOP_WORDS,
                                ngram_range=(1,2))),
        ('clf', MultinomialNB(alpha=1e-2))
    ])

    model.fit(X, y)

    image = "./data/monthy_python.jpg"

    if request.method == 'POST':
        message = request.form['text_message']
        pred = model.predict([message])
    return render_template('predictions.html', data=pred, text_message=message, image=image)



if __name__ == "__main__":
    app.run(debug=True)