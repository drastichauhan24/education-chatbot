# =========================================================
# 🎓 Educational Chatbot - Streamlit Application
# =========================================================

import streamlit as st
import numpy as np
import random
import nltk
nltk.download('punkt_tab')

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_data()
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# 🔹 Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Educational AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ---------------------------------------------------------
# 🔹 Custom CSS (Professional UI)
# ---------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.chat-box {
    max-width: 900px;
    margin: auto;
}
.user-msg {
    background: #0d6efd;
    color: white;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 80%;
    margin-left: auto;
}
.bot-msg {
    background: #e9ecef;
    color: black;
    padding: 12px 16px;
    border-radius: 15px;
    margin: 10px 0;
    width: fit-content;
    max-width: 80%;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🔹 NLTK Setup
# ---------------------------------------------------------
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------
# 🔹 Load Model & Data (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("markchatbot_model_final.h5")
    all_words = joblib.load("all_words.pkl")
    tags = joblib.load("tags.pkl")

    df = pd.read_parquet("train-00000-of-00001.parquet")
    responses_dict = {}

    for tag in tags:
        responses_dict[tag] = df[df["tag"] == tag]["responses"].tolist()

    return model, all_words, tags, responses_dict


model, all_words, tags, responses_dict = load_resources()

# ---------------------------------------------------------
# 🔹 NLP Functions
# ---------------------------------------------------------
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum()]
    return tokens

def bag_of_words(tokens):
    bag = [0] * len(all_words)
    for w in tokens:
        if w in all_words:
            bag[all_words.index(w)] = 1
    return np.array(bag)

def chatbot_response(user_text):
    tokens = preprocess_text(user_text)
    bow = bag_of_words(tokens)
    bow = np.expand_dims(bow, axis=0)

    prediction = model.predict(bow, verbose=0)[0]
    tag = tags[np.argmax(prediction)]

    return random.choice(responses_dict[tag])

# ---------------------------------------------------------
# 🔹 Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎓 Educational Chatbot")
    st.markdown("""
    **Purpose**
    - Student guidance  
    - Academic Q&A  
    - Concept clarification  

    **Built With**
    - TensorFlow  
    - NLP (NLTK)  
    - Streamlit  
    """)

    if st.button("🗑 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

# ---------------------------------------------------------
# 🔹 Session State
# ---------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------------------------------------------------
# 🔹 Main Chat Interface
# ---------------------------------------------------------
st.markdown("<h2 style='text-align:center;'>📘 AI Educational Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask your academic questions below</p>", unsafe_allow_html=True)

st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

for sender, message in st.session_state.chat:
    if sender == "user":
        st.markdown(f"<div class='user-msg'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{message}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 🔹 Input Area
# ---------------------------------------------------------
user_input = st.text_input("💬 Type your question:")

if st.button("Send") and user_input.strip():
    reply = chatbot_response(user_input)

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", reply))

    st.rerun()

# ---------------------------------------------------------
# 🔹 Footer
# ---------------------------------------------------------
st.markdown(
    "<div class='footer'>© 2025 Educational AI Chatbot | Deep Learning Powered</div>",
    unsafe_allow_html=True
)


