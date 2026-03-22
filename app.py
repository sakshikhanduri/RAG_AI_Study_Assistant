import streamlit as st
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📚 AI Study Assistant")

# -------------------- CACHE MODELS --------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatMistralAI(model="mistral-tiny")

embeddings = load_embeddings()
llm = load_llm()

# -------------------- MEMORY --------------------

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )

if "db" not in st.session_state:
    st.session_state.db = None

# -------------------- PDF UPLOAD --------------------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    db_path = f"./chroma_db/{uploaded_file.name}"

    if os.path.exists(db_path):
        st.write("Loading existing database...")
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    else:
        st.write("Processing PDF and creating embeddings...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        db.persist()

    st.session_state.db = db
    st.success("PDF ready! Ask your questions.")

# -------------------- CHAT --------------------

query = st.text_input("Ask a question from your PDF")

if query and st.session_state.db is not None:

    retriever = st.session_state.db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs if doc.page_content])

    # Get chat history
    history = "\n".join(
        [msg.content for msg in st.session_state.memory.chat_memory.messages]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI study assistant. Answer clearly using context and chat history."),
            ("human", "Chat History:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}")
        ]
    )

    final_prompt = prompt.invoke({
        "history": history,
        "context": context,
        "question": query
    })

    # STREAMING RESPONSE (MUST BE INSIDE IF BLOCK)
    st.write("### Assistant:")
    response_text = ""
    placeholder = st.empty()

    with st.spinner("Thinking..."):
        for chunk in llm.stream(final_prompt):
            if chunk.content:
                response_text += chunk.content
                placeholder.markdown(response_text)

    # Save memory
    st.session_state.memory.chat_memory.add_user_message(query)
    st.session_state.memory.chat_memory.add_ai_message(response_text)

# -------------------- CHAT HISTORY --------------------

# if st.session_state.memory:
#     st.write("## Chat History")
#     for msg in st.session_state.memory.chat_memory.messages:
#         st.write(msg.content)