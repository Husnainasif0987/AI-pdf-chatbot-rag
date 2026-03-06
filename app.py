import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

st.title("AI PDF Chatbot (RAG)")

api_key = st.text_input("Enter OpenAI API Key")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    db = FAISS.from_documents(docs, embeddings)

    question = st.text_input("Ask a question from the PDF")

    if question:

        results = db.similarity_search(question)

        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0
        )

        response = llm.predict(str(results))

        st.write("### Answer:")
        st.write(response)
