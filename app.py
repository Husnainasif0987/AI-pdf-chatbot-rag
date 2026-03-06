from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

loader = PyPDFLoader("sample.pdf")
docs = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "What is this document about?"
docs = db.similarity_search(query)

llm = ChatOpenAI()
response = llm.predict(str(docs))

print(response)
