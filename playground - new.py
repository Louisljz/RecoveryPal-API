from langchain_community.document_loaders import CSVLoader
import os
from langchain_community.llms import Clarifai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient("chromadb_sbert")  # Specify data path

client = chromadb.PersistentClient()
print(client.list_collections())
#collection = client.get_collection("f77e8f39-f5f6-4670-ac03-0aae2311acb6", embedding_function=embeddings)


# use model URL
os.environ["CLARIFAI_PAT"]="9a6cd1a04c8c460ba8e27e7db99b4e4e"
MODEL_URL="https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
llm = Clarifai(model_url=MODEL_URL)

query = "Give me a relaxing meditation script."
#docs = collection.similarity_search(query)
#chain = RetrievalQA.from_chain_type(llm=llm,
                                    #chain_type="stuff",
                                    #retriever=collection.as_retriever())

#response = chain(query)
#print(response)