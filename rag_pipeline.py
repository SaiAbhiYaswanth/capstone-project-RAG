from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, LLM_MODEL, TOP_K_RESULTS, RETRIEVAL_POOL
from vector_store import VectorStore
from utils.prompts import SYSTEM_PROMPT
from embeddings import EmbeddingModel

from rank_bm25 import BM25Okapi
import numpy as np

from langgraph.graph import StateGraph
from typing import TypedDict, List


# -----------------------------
# Graph State
# -----------------------------

class RAGState(TypedDict):

    query: str
    vector_docs: List[str]
    keyword_docs: List[str]
    hybrid_docs: List[str]
    final_docs: List[str]
    answer: str


class RAGPipeline:

    def __init__(self):

        self.vector_store = VectorStore()

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL
        )

        self.graph = self.build_graph()

    # ---------------------------------------
    # Vector Search Node
    # ---------------------------------------

    def vector_search_node(self, state: RAGState):

        query = state["query"]

        vector_docs = self.vector_store.search(query, RETRIEVAL_POOL)

        return {"vector_docs": vector_docs}

    # ---------------------------------------
    # Keyword Search (BM25)
    # ---------------------------------------

    def keyword_search(self, query, documents, k=5):

        if len(documents) == 0:
            return []

        tokenized_docs = [doc.lower().split() for doc in documents]

        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)

        ranked_docs = [
            doc for _, doc in sorted(
                zip(scores, documents),
                reverse=True
            )
        ]

        return ranked_docs[:k]

    def keyword_search_node(self, state: RAGState):

        query = state["query"]

        all_docs = self.vector_store.get_all_documents()

        keyword_docs = self.keyword_search(query, all_docs, RETRIEVAL_POOL)

        return {"keyword_docs": keyword_docs}

    # ---------------------------------------
    # Hybrid Merge Node
    # ---------------------------------------

    def hybrid_node(self, state: RAGState):

        vector_docs = state["vector_docs"]
        keyword_docs = state["keyword_docs"]

        hybrid_docs = list(set(vector_docs + keyword_docs))

        return {"hybrid_docs": hybrid_docs}

    # ---------------------------------------
    # Reranking using embeddings
    # ---------------------------------------

    def rerank(self, query, docs, k=4):

        if len(docs) == 0:
            return []

        model = EmbeddingModel.load_model()

        query_embedding = model.encode(query)

        doc_embeddings = model.encode(docs)

        similarities = np.dot(doc_embeddings, query_embedding)

        ranked = [
            doc for _, doc in sorted(
                zip(similarities, docs),
                reverse=True
            )
        ]

        return ranked[:k]

    def rerank_node(self, state: RAGState):

        query = state["query"]
        docs = state["hybrid_docs"]

        final_docs = self.rerank(query, docs, TOP_K_RESULTS)

        return {"final_docs": final_docs}

    # ---------------------------------------
    # LLM Answer Node
    # ---------------------------------------

    def generate_answer_node(self, state: RAGState):

        query = state["query"]
        final_docs = state["final_docs"]

        if len(final_docs) == 0:
            return {
                "answer": "The knowledge base is empty. Please upload a PDF document so I can answer your questions."
            }

        context = ""

        for i, doc in enumerate(final_docs, 1):
            context += f"\nChunk {i}:\n{doc}\n"

        prompt = f"""
{SYSTEM_PROMPT}

DOCUMENT CONTEXT:
-----------------
{context}

USER QUESTION:
--------------
{query}

INSTRUCTIONS:
-------------
Answer the user's question using ONLY the document context above.

Do not explain how you found the answer.
Do not mention chunks or context analysis.

Provide a clear and detailed explanation.

If the document contains:
• definitions
• explanations
• examples
• properties
• related concepts

include them in your answer.

If the document contains:
- tables
- diagrams
- truth tables
- lists of operators
- examples

present them in structured format.

Use paragraphs or bullet points when appropriate.
Make the explanation easy to understand for a student.
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return {"answer": response.content}

    # ---------------------------------------
    # Build LangGraph
    # ---------------------------------------

    def build_graph(self):

        builder = StateGraph(RAGState)

        builder.add_node("vector_search", self.vector_search_node)
        builder.add_node("keyword_search", self.keyword_search_node)
        builder.add_node("hybrid", self.hybrid_node)
        builder.add_node("rerank", self.rerank_node)
        builder.add_node("generate_answer", self.generate_answer_node)

        # Entry node (replacement for START)
        builder.set_entry_point("vector_search")

        builder.add_edge("vector_search", "keyword_search")
        builder.add_edge("keyword_search", "hybrid")
        builder.add_edge("hybrid", "rerank")
        builder.add_edge("rerank", "generate_answer")

        # Finish node (replacement for END)
        builder.set_finish_point("generate_answer")

        return builder.compile()

    # ---------------------------------------
    # Run Pipeline
    # ---------------------------------------

    def answer_question(self, query):

        result = self.graph.invoke({"query": query})

        return result["answer"]