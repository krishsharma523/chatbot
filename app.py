import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="Hugging Face RAG Chatbot", layout="centered")
st.title("🤗 Hugging Face RAG Chatbot")
st.write("Upload a document (PDF, DOCX, TXT) and ask questions using a cloud-hosted LLM from Hugging Face.")

# Load Hugging Face API key
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📁 Upload a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        file_path = tmp.name

    ext = uploaded_file.name.split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file format.")
        st.stop()

    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever()

    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512}, huggingfacehub_api_token=hf_token)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ File processed. Ask your questions below!")

if st.session_state.qa_chain:
    query = st.text_input("💬 Ask a question")
    if query:
        result = st.session_state.qa_chain.run(query)
        st.write("🤖 Answer:")
        st.success(result)