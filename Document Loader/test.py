from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
data = PyPDFLoader("Document Loader/Machine_Learning.pdf")
response = data.load()
splitter = CharacterTextSplitter(separator= " ", chunk_size= 3000, chunk_overlap=1)
token_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=10)
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=10)

chunks_text = splitter.split_documents(response)
chunks_tokens = token_splitter.split_documents(response)
chunks_recursive = recursive_splitter.split_documents(response)

print(chunks_text[0].page_content) #separates on basis of \n\n
print(len(chunks_text))
print(chunks_tokens[0].page_content) #separates on basis of tokens
print(len(chunks_tokens))
print(chunks_recursive[0].page_content) #separates on basis of \n\n, \n, " ", and if the chunk is still too big, it will split on the basis of characters
print(len(chunks_recursive))