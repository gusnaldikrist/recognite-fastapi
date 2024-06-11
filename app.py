import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
import tensorflow as tf
import string
import re
import os

app = FastAPI()

# Load and preprocess dataset from local file
data_path = 'dataset/dataset.csv'  # Update this path
df = pd.read_csv(data_path)

# Ensure 'sentences' column exists
if 'sentences' not in df.columns:
    raise KeyError("'sentences' column not found in the dataset")

# Clean the dataset text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(char for char in text if ord(char) < 128)
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

df['sentences'] = df['sentences'].apply(clean_text)

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(df['sentences'].values)

# Ensure padding length matches model input shape
maxlen = 186  # Example length, adjust based on your model's input shape

def preprocess_text(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    return padded

@app.post('/predict')
def predict(text: str = Form(...)):
    try:
        processed_text = preprocess_text(text)
        model = tf.keras.models.load_model('model/model.h5')
        predictions = model.predict(processed_text)
        sentiment = np.argmax(predictions)
        probability = float(max(predictions[0]))
        sentiment_labels = ['Label 0', 'Label 1', 'Label 2', 'Label 3']  # Update these labels as per your model
        predicted_label = sentiment_labels[sentiment]
        
        return {
            "Text": text,
            "Sentiment": predicted_label,
            "Probability": probability
        }
    except Exception as e:
        return {"error": str(e)}

