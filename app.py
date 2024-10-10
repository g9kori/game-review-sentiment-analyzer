import streamlit as st
import joblib
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model = joblib.load('sentiment_model.pkl')
tokenizer = joblib.load('tokenizer.pkl')

# Function to classify sentiment
def classify_sentiment(review):
    input_text = f"Classify the sentiment of this review: {review}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids)
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

# Streamlit app layout
st.title("Sentiment Analysis for Game Reviews")
st.write("Enter a game review to classify its sentiment:")

# Text input for user to enter the review
review_text = st.text_area("Review Text")

if st.button("Classify Sentiment"):
    if review_text:
        sentiment = classify_sentiment(review_text)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to classify.")
