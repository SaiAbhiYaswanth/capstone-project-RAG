import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_DB_PATH = "knowledge_base"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

TOP_K_RESULTS = 4
RETRIEVAL_POOL = 10