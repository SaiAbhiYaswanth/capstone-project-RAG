import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME
from embeddings import EmbeddingModel
import uuid


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    # ---------------------------
    # Add documents
    # ---------------------------

    def add_documents(self, chunks, metadata):

        embeddings = EmbeddingModel.embed(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata
        )

    # ---------------------------
    # Vector Search
    # ---------------------------

    def search(self, query, k=4):

        count = self.collection.count()

        if count == 0:
            return []

        embedding = EmbeddingModel.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        return results["documents"][0]

    # ---------------------------
    # Get all documents
    # ---------------------------

    def get_all_documents(self):

        results = self.collection.get()

        if "documents" not in results:
            return []

        return results["documents"]