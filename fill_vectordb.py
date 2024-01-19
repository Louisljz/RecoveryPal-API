from pinecone import Pinecone as pinecone_client
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

from dotenv import load_dotenv
import os

load_dotenv('API/.env')

loader = CSVLoader(file_path="meditation-scripts.csv", source_column="Text")
docs = loader.load()

pc = pinecone_client(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('meditation-scripts')

vectorstore = Pinecone(index, HuggingFaceEmbeddings(), "text")
vectorstore.add_documents(docs)
print('Data successfully pushed to Pinecone')