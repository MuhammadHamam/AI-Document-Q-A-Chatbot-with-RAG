import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_split_pdf, create_vector_store, create_qa_chain, answer_question

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="centered")

st.title("📄 AI Document Q&A Chatbot")
st.markdown("Upload a PDF and ask questions about it using AI.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                chunks = load_and_split_pdf(tmp_path)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                os.unlink(tmp_path)
            st.success(f"Ready! {len(chunks)} chunks indexed.")

    if st.session_state.doc_name:
        st.info(f"Active: {st.session_state.doc_name}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if st.session_state.qa_chain is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.caption(f"Sources: pages {msg['sources']}")

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(
                    st.session_state.qa_chain, question
                )
            st.write(answer)
            if sources:
                st.caption(f"Sources: pages {sources}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

st.markdown("---")
st.caption("Built with LangChain + FAISS + Google Gemini | RAG Pipeline")
