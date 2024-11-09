# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import torch
import os
from transformers import BertTokenizer, BertModel
from word2number import w2n
import spacy

# Load necessary components
app = Flask(__name__)

# Load data and model files
filename = 'sentiment_model.pkl'
loaded_data = joblib.load(filename)
model = loaded_data['model']
tokenizer = loaded_data['tokenizer']
bert_model = loaded_data['bert_model']
max_len = loaded_data['max_len']
nlp = spacy.load("en_core_web_sm")

# Function Definitions
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def get_bert_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def parse_numerical_info(sentence):
    doc = nlp(sentence)
    numerical_info = []
    for token in doc:
        if token.like_num:
            try:
                num = w2n.word_to_num(token.text)
                numerical_info.append(num)
            except ValueError:
                continue
    return np.array(numerical_info) if numerical_info else np.zeros(3)

def combine_features(bert_embedding, numerical_values, aspect):
    if len(numerical_values) < 3:
        numerical_values = np.pad(numerical_values, (0, 3 - len(numerical_values)), 'constant')
    aspect_embedding = get_bert_embeddings([aspect], tokenizer, bert_model).flatten()
    combined = np.concatenate([bert_embedding, numerical_values, aspect_embedding])
    combined = combined[:max_len]
    if len(combined) < max_len:
        combined = np.pad(combined, (0, max_len - len(combined)), 'constant')
    return combined

def predict_sentiment(sentence, aspect):
    preprocessed_sentence = preprocess_text(sentence)
    bert_embedding = get_bert_embeddings([preprocessed_sentence], tokenizer, bert_model).flatten()
    numerical_info = parse_numerical_info(preprocessed_sentence)
    combined_features = combine_features(bert_embedding, numerical_info, aspect)
    features = combined_features.reshape(1, -1)
    prediction = model.predict(features).flatten()
    return prediction[0]

# Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load data and extract row
            data = pd.read_csv('input.csv')
            row_number = int(request.form.get('Rownumber'))
            sentence = data.iloc[row_number, 0]
            aspect = data.iloc[row_number, 1]
            predicted_score = predict_sentiment(sentence, aspect)

            # Render result
            return render_template('result.html', prediction=predicted_score, sentence=sentence, aspect=aspect)
        except Exception as e:
            return render_template('result.html', error=f"Error: {str(e)}")

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
