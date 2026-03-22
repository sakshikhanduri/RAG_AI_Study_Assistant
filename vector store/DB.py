from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is a high-level programming language. It is highly used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in Deep Learning.", metadata={"source": "DL_book"}),
]

embedding = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)

result = vector_store.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vector_store.as_retriever()
docs = retriever.invoke("Explain deep learning.")
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
