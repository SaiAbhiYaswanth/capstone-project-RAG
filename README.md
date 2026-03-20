# 🤖 Intelligent RAG-Based Doubt Solver Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot that answers user queries from uploaded PDF documents using hybrid retrieval (vector search + BM25), embedding-based reranking, and Groq LLaMA 3.1.

---

## 🚀 Overview

This project is an intelligent document-based chatbot built using Streamlit, LangChain, and LangGraph. Users can upload PDF documents, and the system builds a knowledge base to answer questions accurately using a hybrid retrieval pipeline.

Unlike basic RAG systems, this implementation combines vector search, keyword search (BM25), and reranking to improve retrieval quality and response accuracy.

---

## ✨ Key Features

* 📄 Upload PDF documents dynamically
* 🧠 Build a persistent knowledge base using ChromaDB
* 🔍 Hybrid Retrieval:

  * Vector similarity search
  * Keyword-based BM25 search
* 🎯 Embedding-based reranking for better relevance
* 🤖 LLM-powered answers using Groq (LLaMA 3.1)
* 🔄 Multi-chat session management
* ⚡ Fast UI using Streamlit with caching

---

## 🧠 System Architecture

The system follows a **multi-stage RAG pipeline** implemented using LangGraph:

1. **PDF Processing**

   * Extract text using PyPDF
   * Normalize table-like structures
   * Split text into chunks using RecursiveCharacterTextSplitter

2. **Embedding & Storage**

   * Convert chunks into embeddings using SentenceTransformers
   * Store in ChromaDB vector database

3. **Query Processing Pipeline**

   * User query enters LangGraph pipeline
   * Vector Search retrieves top candidates
   * BM25 Keyword Search retrieves additional relevant documents
   * Hybrid merge combines both results
   * Embedding-based reranking selects top results

4. **Answer Generation**

   * Selected chunks are passed as context
   * Groq LLaMA 3.1 generates the final answer

---

## 🔄 RAG Pipeline Flow

User Query
→ Vector Search (ChromaDB)
→ Keyword Search (BM25)
→ Hybrid Merge
→ Reranking (Embeddings)
→ LLM Response (Groq)

---

## 🛠️ Tech Stack

| Component       | Technology                  |
| --------------- | --------------------------- |
| Frontend        | Streamlit                   |
| LLM             | Groq (LLaMA 3.1-8B Instant) |
| Framework       | LangChain, LangGraph        |
| Vector Database | ChromaDB                    |
| Embeddings      | SentenceTransformers        |
| Retrieval       | BM25 (rank-bm25)            |
| PDF Processing  | PyPDF                       |

---

## 📁 Project Structure

```
capstone-project-RAG/
│
├── app.py              # Streamlit UI and chat interface
├── rag_pipeline.py     # LangGraph-based RAG pipeline
├── vector_store.py     # ChromaDB vector database operations
├── embeddings.py       # Embedding model wrapper
├── pdf_loader.py       # PDF loading and chunking
├── config.py           # Configuration and environment variables
├── requirements.txt    # Dependencies
├── utils/
│   └── prompts.py      # System prompt for LLM
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SaiAbhiYaswanth/capstone-project-RAG.git
cd capstone-project-RAG
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🧪 How to Use

1. Open the Streamlit app
2. Upload a PDF document
3. Wait for it to be processed and stored
4. Ask questions related to the document
5. Get accurate, context-based answers

---

## ⚡ Performance Optimizations

* Streamlit caching (`@st.cache_resource`) for:

  * RAG pipeline
  * Vector store
* Persistent storage using ChromaDB
* Avoids reprocessing already uploaded PDFs

---

## 🔒 Security

* API keys are managed using `.env`
* `.gitignore` prevents sensitive data from being pushed

---

## 📌 Future Improvements

* 🌐 Web search integration (Hybrid RAG + Web)
* 📊 Source citation display
* 🧩 Improved UI/UX
* 🔐 User authentication system
* 📂 Multi-document filtering

---

## 👨‍💻 Author

**Aravapalli Sai Abhi Yaswanth**

---

## ⭐ Support
If you find this project useful, consider giving it a ⭐ on GitHub!
