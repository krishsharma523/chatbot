# 🤗 Hugging Face RAG Chatbot

This is a simple Retrieval-Augmented Generation (RAG) chatbot app built with Streamlit, LangChain, and Hugging Face LLMs.

## 🔧 Setup

1. Clone the repo:
```
git clone https://github.com/YOUR_USERNAME/huggingface-rag-chatbot.git
cd huggingface-rag-chatbot
```

2. Add your Hugging Face API token:
Create a file `.streamlit/secrets.toml` with the following:

```
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token_here"
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
streamlit run app.py
```

## 📄 Features

- Upload PDF, DOCX, or TXT files
- Ask questions about document content
- Uses `google/flan-t5-large` from Hugging Face Hub

## ☁️ Deploy

You can deploy this to [Streamlit Cloud](https://streamlit.io/cloud) easily by linking your GitHub repo and setting the `HUGGINGFACEHUB_API_TOKEN` in the secrets UI.

---