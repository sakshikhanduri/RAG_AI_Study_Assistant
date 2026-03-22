from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("Document Loader/Machine_Learning.pdf")
response = data.load()
print(response[0].page_content)
print(len(response))