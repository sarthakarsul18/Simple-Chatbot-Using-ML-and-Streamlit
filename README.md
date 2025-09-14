# 💬 Streamlit Chatbot

A simple **Machine Learning powered chatbot** built with **Streamlit**, **NLTK**, and **Scikit-learn**.  
This chatbot classifies user input into predefined intents using **TF-IDF Vectorization** and **Logistic Regression**.

---

## 🚀 Features
- 🗨️ Interactive chat interface with **Streamlit**  
- 📖 Maintains full conversation history  
- 🔍 Preprocessing with **lemmatization, lowercase, punctuation removal** for better accuracy  
- ⚡ Fallback responses when input is not recognized  
- 🎯 Uses **TF-IDF + Logistic Regression** for intent classification  
- 📂 Easily extendable by adding new intents  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Streamlit** (UI)  
- **NLTK** (text preprocessing)  
- **Scikit-learn** (ML model: Logistic Regression)  

---

## 📦 Installation

1. Clone the repository  
```bash
git clone https://github.com/sarthakarsul18/Simple-Chatbot-Using-ML-and-Streamlit
cd Simple-Chatbot-Using-ML-and-Streamlit
```

## Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

## Install dependencies
```
pip install -r requirements.txt
```
## ▶️ Usage

Run the chatbot with Streamlit:
```
streamlit run chatbot.py
```

Then open the link shown in your terminal (usually http://localhost:8501) in your browser.

### 📝 Example Conversation
You: Hi
Bot: Hello!

You: What can you do?
Bot: I am a chatbot. My purpose is to assist you.

You: Thanks
Bot: You're welcome!

You: Bye
Bot: Goodbye!

### 📚 Adding New Intents

To add new intents:

Open chatbot.py

Find the intents list

Add a new dictionary with tag, patterns, and responses

Example:

{
  "tag": "joke",
  "patterns": ["Tell me a joke", "Make me laugh"],
  "responses": ["Why don't scientists trust atoms? Because they make up everything!"]
}


Then restart the app with:

streamlit run chatbot.py

## ✅ Requirements

Python 3.8+

Streamlit

NLTK

Scikit-learn

You can install them with:
```
pip install streamlit nltk scikit-learn
```


## 👨‍💻 Author

Sarthak Arsul
