import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

load_dotenv()

def get_groq_key():
    try:
        return st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=get_groq_key(),
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant answering questions based on the provided document.
    Use only the information from the context below to answer.
    If the answer is not in the context, say "I couldn't find that in the document."

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def answer_question(qa_chain_tuple, question):
    chain, retriever = qa_chain_tuple
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = sorted(list(set([
        doc.metadata.get("page", 0) + 1
        for doc in docs
    ])))
    return answer, sources




