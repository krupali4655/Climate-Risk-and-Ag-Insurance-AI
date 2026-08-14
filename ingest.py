import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 Starting Offline Data Ingestion Pipeline...")

# 1. Load all PDFs from the 'data' directory
loader = PyPDFDirectoryLoader("./data")
docs = loader.load()
print(f"📄 Loaded {len(docs)} pages from the data folder.")

# 2. Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split documents into {len(chunks)} chunks.")

# 3. Embed and save to a PERSISTENT Chroma database on your hard drive
embeddings = FastEmbedEmbeddings()
# Notice we add 'persist_directory' so it saves to a folder named 'chroma_db'
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("✅ Ingestion complete! Vector database saved to ./chroma_db")