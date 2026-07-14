import os

import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from rag_logic import build_grounded_prompt, should_answer_from_context

# ----------------------------
# Load Environment Variables
# ----------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #fef08a 0%, #facc15 100%);
        border: 1px solid #f59e0b;
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.16);
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #134e4a;
        font-size: 0.98rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: #4d7c0f;
        border: 1px solid #84cc16;
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        box-shadow: 0 4px 12px rgba(132, 204, 22, 0.16);
    }
    .metric-card b {
        color: #fefce8;
    }
    .metric-card, .metric-card p {
        color: #ecfccb;
    }
    .answer-box {
        background: #4d7c0f;
        border: 1px solid #84cc16;
        border-left: 4px solid #bef264;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-top: 0.8rem;
        color: #f8fafc;
    }
    div[data-testid="stTextInput"] > div > div > input {
        border-radius: 10px;
        border: 1px solid #84cc16;
        background-color: #fef9c3;
        color: #134e4a;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] > div > div > input::placeholder,
    .stMarkdown, .stText {
        color: #134e4a !important;
    }
    .sidebar .block-container {
        background: #dcfce7;
    }
    .sidebar .stTextArea, .sidebar .stTextInput {
        border: 1px solid #4ade80;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🕘 Recent Questions")
st.sidebar.caption("Your latest searches")

if st.sidebar.button("Clear history", use_container_width=True):
    st.session_state.history = []
    st.sidebar.success("History cleared")

if st.session_state.history:
    for item in reversed(st.session_state.history[-8:]):
        st.sidebar.markdown(f"- {item}")
else:
    st.sidebar.info("No questions yet")

st.markdown(
    """
    <div class="hero-card" style="margin-top: 0.4rem; margin-bottom: 1.2rem;">
        <div class="hero-title">📄 PDF Assistant</div>
        <p class="hero-subtitle">Ask questions about your uploaded PDF and receive answers grounded only in the document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Embedding Model
# ----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# Connect Pinecone
# ----------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ----------------------------
# Gemini
# ----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# User Question
# ----------------------------

question = st.text_input("Ask a question about the PDF", placeholder="e.g. What are the key findings?")

if question and question.strip():
    question_text = question.strip()
    if not st.session_state.history or st.session_state.history[-1] != question_text:
        st.session_state.history.append(question_text)

    try:
        st.write("✅ Question received")

        docs = retriever.invoke(question)
        st.write(f"✅ Retrieved {len(docs)} documents")

        context = "\n\n".join([doc.page_content for doc in docs])
        st.write("✅ Context created")

        if not should_answer_from_context(question, context):
            st.markdown("<div class='answer-box'><b>Answer</b><br>I couldn't find the answer in the uploaded PDF.</div>", unsafe_allow_html=True)
            st.stop()

        prompt = build_grounded_prompt(question, context)

        st.write("✅ Sending to Gemini...")

        response = llm.invoke(prompt)

        st.write("✅ Gemini responded!")

        st.markdown("<div class='answer-box'><b>Answer</b><br>" + response.content.replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")