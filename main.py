import streamlit as st
import re
import string

# For demonstration: a dummy prediction function
def predict_fake_news(text):
    # Dummy rule: if 'fake' in text, it's fake news
    if 'fake' in text.lower():
        return "Fake News ❌"
    else:
        return "Real News ✅"

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

# Streamlit App
st.title("📰 Fake News Detector")

user_input = st.text_area("Paste your news content here:")

if st.button("Check"):
    cleaned = clean_text(user_input)
    result = predict_fake_news(cleaned)
    st.subheader("Result:")
    st.write(result)
