from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
url = "https://www.apple.com/in/macbook-neo/"

data = WebBaseLoader(url)
response = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant which summarizes the text."),
    ("human", "{data}")
])

model = ChatMistralAI(model="mistral-small-2506")

prompt = template.format_prompt(data=response)
result = model.invoke(prompt)
print(result.content)