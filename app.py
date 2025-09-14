import os
import ssl
import nltk
import streamlit as st
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer

# SSL & NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello!", "Hey!", "I'm fine, thank you.", "Nothing much!"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot.", "My purpose is to assist you.", "I can answer questions and provide assistance."]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, track your income and expenses, allocate funds for essentials, savings, and discretionary spending.", "Use the 50/30/20 rule: 50% essentials, 30% discretionary, 20% savings."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score represents your creditworthiness. Higher scores increase chances of loan approval.", "You can check your credit score on websites like Credit Karma or Credit Sesame."]
    }
]

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Prepare data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(preprocess(pattern))
        tags.append(intent['tag'])

# Train model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text_proc = preprocess(input_text)
    input_vec = vectorizer.transform([input_text_proc])
    tag = clf.predict(input_vec)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that. Can you rephrase?"

# Streamlit app
def main():
    st.title("💬 Chatbot")
    st.write("Welcome! Type a message below to start chatting.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

    for msg in st.session_state.history:
        st.write(msg)

    if st.session_state.history and st.session_state.history[-1].startswith("Bot: Goodbye"):
        st.write("Thank you for chatting! Have a great day.")
        st.stop()

if __name__ == "__main__":
    main()
