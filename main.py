import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = vector_store.as_retriever(
    search_type="similarity",
   search_kwargs={"k": 4} 
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant. USe ONLY the provided context to answeer the question. If the answer is not in the context, say "I could not find the answer in the document." """),
        ("human", """Context:{context}
         Question: {question}""")
    ]
)

print("RAG system created successfully!")
print("Press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        print("Exiting...")
        break

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)
    print("Assistant:", response.content)