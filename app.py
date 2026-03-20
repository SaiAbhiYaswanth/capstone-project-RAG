import streamlit as st
import uuid
from rag_pipeline import RAGPipeline
from pdf_loader import PDFLoader
from vector_store import VectorStore

st.set_page_config(page_title="AI Doubt Solver", layout="wide")

# ---------------------------------------------------
# CACHE HEAVY OBJECTS (prevents reload on rerun)
# ---------------------------------------------------

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

@st.cache_resource
def load_vector_store():
    return VectorStore()

pipeline = load_pipeline()
vector_store = load_vector_store()

st.title("Intelligent Doubt-Solver Chatbot")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# Track uploaded PDFs to avoid reprocessing
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("💬 Chats")

# New Chat
if st.sidebar.button("➕ New Chat"):
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = []
    st.session_state.current_chat = new_chat_id

st.sidebar.divider()

# ---------------------------------------------------
# PDF Upload
# ---------------------------------------------------

st.sidebar.subheader("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload document",
    type=["pdf"]
)

if uploaded_file:

    # prevent reprocessing same file
    if uploaded_file.name not in st.session_state.uploaded_files:

        text = PDFLoader.load_pdf(uploaded_file)

        chunks = PDFLoader.chunk_text(text)

        metadata = [{"source": uploaded_file.name}] * len(chunks)

        vector_store.add_documents(chunks, metadata)

        st.session_state.uploaded_files.add(uploaded_file.name)

        st.sidebar.success("PDF added to knowledge base!")

st.sidebar.divider()

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

st.sidebar.subheader("History")

for chat_id, messages in st.session_state.chats.items():

    if len(messages) > 0:
        label = messages[0]["content"][:40]
    else:
        label = "New Chat"

    if st.sidebar.button(label, key=chat_id):
        st.session_state.current_chat = chat_id

# ---------------------------------------------------
# CHAT AREA
# ---------------------------------------------------

messages = st.session_state.chats[st.session_state.current_chat]

chat_container = st.container()

with chat_container:

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------------------------------------
# INPUT BAR
# ---------------------------------------------------

user_query = st.chat_input("Ask your question...")

# ---------------------------------------------------
# HANDLE QUERY
# ---------------------------------------------------

if user_query:

    messages.append({"role": "user", "content": user_query})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_query)

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = pipeline.answer_question(user_query)
                st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})