# AI Study Assistant – PDF Chatbot (RAG)

This project is an AI-powered Study Assistant that allows users to upload PDF books/notes and ask questions from them. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers from the uploaded documents.

## Features
- Upload PDF documents
- Ask questions from PDF
- Conversational chat with memory
- Fast semantic search using vector database
- Streaming responses
- Multi-PDF support
- Persistent vector database

## Tech Stack
- Streamlit (UI)
- LangChain (RAG Pipeline)
- ChromaDB / FAISS (Vector Database)
- HuggingFace Embeddings
- Mistral LLM API
- Python

## How It Works
1. User uploads PDF
2. PDF is split into chunks
3. Embeddings are created
4. Stored in Vector Database
5. User asks question
6. Relevant chunks retrieved
7. Sent to LLM
8. LLM generates answer

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py