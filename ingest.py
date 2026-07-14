import os
from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# -------------------------
# Load PDF
# -------------------------
loader = PyPDFLoader("uploaded.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# -------------------------
# Split into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# -------------------------
# Hugging Face Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded.")

# -------------------------
# Connect to Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# -------------------------
# Upload to Pinecone
# -------------------------
vectorstore.add_documents(chunks)

print("✅ PDF successfully uploaded to Pinecone!")