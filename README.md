# AI Document Q&A Chatbot with RAG

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-purple)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

An AI-powered chatbot that lets you upload any PDF and ask
questions about it using a RAG (Retrieval Augmented Generation)
pipeline the same architecture used in enterprise AI systems.

**[Live Demo](https://ai-document-q-a-chatbot-with-rag-tfpturo3xuxoua63sxsbtd.streamlit.app/)**

---

## How It Works

PDF Upload
↓
Split into chunks (1000 chars, 200 overlap)
↓
Embed chunks locally → FAISS Vector Store
↓
User asks question
↓
Retrieve top 4 relevant chunks via similarity search
↓
Send chunks + question to LLaMA 3.3 via Groq API
↓
Return answer + source page numbers

---

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — vector similarity search (Facebook AI)
- **HuggingFace Sentence Transformers** — local text embeddings
- **Groq + LLaMA 3.3 70B** — LLM for answer generation
- **PyPDF** — PDF loading and parsing
- **Streamlit** — chat interface and deployment

---

## Key Concepts Demonstrated

- RAG (Retrieval Augmented Generation) pipeline
- Vector embeddings and similarity search with FAISS
- Prompt engineering for factual document Q&A
- Chunk size and overlap optimization
- LLM integration via LangChain and Groq API
- Session state management in Streamlit
- Local embeddings vs API embeddings trade-offs

---

## How to Run Locally

```bash
git clone https://github.com/MuhammadHamam/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt

# Create .env file with your Groq API key:
echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app.py
```

Get a free Groq API key at: console.groq.com

---

## Deploying on Streamlit Cloud

1. Push repo to GitHub (without .env)
2. Go to share.streamlit.io
3. Select your repo and set main file to app.py
4. Go to App Settings → Secrets and add:
5. Click Save and deploy

---

## Project Structure

rag-chatbot/
├── app.py              # Streamlit chat interface
├── rag_pipeline.py     # RAG logic (chunking, embedding, retrieval)
├── requirements.txt    # Dependencies
├── .gitignore          # Excludes .env from git
└── README.md

---

## Author

**Muhammad Hamam** — Data Scientist & AI Engineer
[LinkedIn](https://www.linkedin.com/in/muhammad-hamam-b90455374/) | [GitHub](https://github.com/MuhammadHamam)
