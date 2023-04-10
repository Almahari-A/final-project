import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except Exception as e:
        return None

from flask import Flask, request, jsonify, render_template
from joblib import load
from keras_preprocessing.text import tokenizer_from_json
from tensorflow import keras 
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the label encoder for the LSTM model
label_encoder = load('lstm_label_encoder.joblib')

# Create the id_to_category dictionary
id_to_category = {i: c for i, c in enumerate(label_encoder.classes_)}

# Load the CNN model
cnn_model = load_model('presence_classifier_cnn_tf.keras')

# Load the LSTM model
lstm_model = load_model('lstm_category_classifier.h5')

# Load the LSTM tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_data = f.read()
    lstm_tokenizer = tokenizer_from_json(tokenizer_data)

# Load the CNN tokenizer
cnn_tokenizer = load('presence_tokenizer.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']

    # Extract text from the URL
    text = extract_text_from_url(url)
    if not text:
        return jsonify({"error": "Failed to extract text from the provided URL"})

    def preprocess_text(text, tokenizer, maxlen):
        tokenized_text = tokenizer.texts_to_sequences([text])
        padded_text = pad_sequences(tokenized_text, maxlen=maxlen)
        return padded_text

    # Preprocess text for both models
    cnn_input = preprocess_text(text, cnn_tokenizer, 100)
    lstm_input = preprocess_text(text, lstm_tokenizer, 250)

    # Make predictions
    presence_pred = cnn_model.predict(cnn_input)
    presence_class = "Dark" if presence_pred > 0.5 else "Not Dark"

    category_pred = lstm_model.predict(lstm_input)
    category_class = id_to_category[np.argmax(category_pred)]

    # Return predictions as JSON
    response = {
        'presence': presence_class,
        'category': category_class
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()




