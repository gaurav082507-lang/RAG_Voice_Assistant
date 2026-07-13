import os
import re
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from project import build_retriever, transcribe_all, audio_chunks, text_to_speech


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VoiceDoc AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling (dark theme, purple gradient title, pill badges)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
    }
    .eyebrow {
        text-align: center;
        letter-spacing: 6px;
        color: #7dd3fc;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .hero-title {
        text-align: center;
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(90deg, #7c3aed, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 17px;
        max-width: 720px;
        margin: 0 auto 28px auto;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 8px 18px;
        color: #93c5fd;
        font-size: 14px;
    }
    .hint-box {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 22px;
        color: #d1d5db;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1420;
        border-right: 1px solid #1f2937;
    }
    .status-pill {
        background-color: #052e1b;
        border: 1px solid #14532d;
        color: #4ade80;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_for_tts(text: str) -> str:
    text = re.sub(r"[#*_`]", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


SYSTEM_PROMPT = """You are a helpful voice assistant that answers user questions using the provided context (retrieved documents). Your response will be converted to speech, so you must write in plain, natural spoken language only.

Formatting rules:
1. Do not use any markdown symbols such as hashtags, asterisks, underscores, bullet points, numbered lists, or headers.
2. Do not use special characters for emphasis or structure of any kind.
3. Write in complete, natural sentences and full paragraphs, the way a person would speak out loud.
4. If you need to list multiple points, say them as a flowing sentence using words like first, second, next, additionally, or finally, instead of using list symbols or line breaks.
5. Keep the tone conversational and clear, as if you are explaining the answer to someone listening, not reading a document.

Answering rules:
6. Always try to answer the user's question using only the information in the context section below.
7. Give a detailed, thorough, and complete answer. Do not give short or overly brief responses.
8. If the context contains sufficient information to answer the question, answer using that context directly and do not mention anything about the context being insufficient.
9. If the context does not contain enough information to answer the question, partially or fully, you may use your own general knowledge to answer.
10. In that case, you must clearly state at the start of your answer, in plain spoken language, that the provided documents did not contain this information and that you are answering from your own general knowledge.
11. Never blend unmarked outside knowledge into an answer that appears to come from the context.
12. Do not fabricate or guess citations, page numbers, or sources from the context.
13. Avoid unnecessary repetition, but do not sacrifice completeness or detail for the sake of brevity.

Context:
{context}
"""

template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "This is my question {user_question}"),
    ]
)


def get_context(docs, query):
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context, "user_question": query}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "answer_audio_path" not in st.session_state:
    st.session_state.answer_audio_path = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ VoiceDoc AI")
    st.caption("PDF RAG · FAISS/MMR retrieval · Mistral · Voice in, voice out")

    api_key_present = bool(os.getenv("MISTRAL_API_KEY"))
    if api_key_present:
        st.markdown('<div class="status-pill">● Mistral API key connected</div>', unsafe_allow_html=True)
    else:
        st.error("MISTRAL_API_KEY not found in environment (.env)")

    st.markdown("---")
    st.markdown("#### 📄 Upload Document")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    st.markdown("---")
    st.markdown("#### 🎤 Ask by Voice")
    st.caption("Record your question, or upload an audio file instead.")
    recorded_audio = st.audio_input("Record your question")
    uploaded_audio = st.file_uploader(
        "...or upload an audio file", type=["wav", "mp3", "m4a", "ogg"]
    )

    st.markdown("---")
    run_clicked = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main hero section
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">VOICE POWERED DOCUMENT ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">VoiceDoc AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload a PDF and ask your question by voice. '
    "Get an answer grounded in your document, read back to you, in seconds.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="badge-row">
        <div class="badge">📄 PDF Retrieval</div>
        <div class="badge">🧩 FAISS + MMR</div>
        <div class="badge">🔗 LangChain</div>
        <div class="badge">🤖 Mistral</div>
        <div class="badge">🎙️ Voice In / Out</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run_clicked:
    st.markdown(
        '<div class="hint-box">👉 Upload a PDF and record or upload your question in the sidebar, '
        "then click <b>Run Analysis</b> to get started.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked:
    if pdf_file is None:
        st.error("Please upload a PDF document first.")
        st.stop()

    if recorded_audio is None and uploaded_audio is None:
        st.error("Please record your question or upload an audio file first.")
        st.stop()

    work_dir = tempfile.mkdtemp(prefix="voicedoc_")

    # Save PDF to disk
    pdf_path = os.path.join(work_dir, pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Build retriever only if a new PDF was uploaded
    if st.session_state.pdf_name != pdf_file.name or st.session_state.retriever is None:
        with st.spinner("Reading and indexing the document..."):
            st.session_state.retriever = build_retriever(pdf_path)
            st.session_state.pdf_name = pdf_file.name

    # Save audio (recorded takes priority over uploaded)
    audio_source = recorded_audio if recorded_audio is not None else uploaded_audio
    audio_ext = ".wav" if recorded_audio is not None else os.path.splitext(uploaded_audio.name)[1]
    audio_path = os.path.join(work_dir, f"question{audio_ext}")
    with open(audio_path, "wb") as f:
        f.write(audio_source.getbuffer())

    with st.spinner("Transcribing your question..."):
        chunk_audio_path = audio_to_chunks(audio_path, output_dir=os.path.join(work_dir, "chunks"))
        query = transcribe_all(chunk_audio_path=chunk_audio_path)
        st.session_state.transcript = query

    with st.spinner("Thinking through your document..."):
        parser = StrOutputParser()
        llm = ChatMistralAI(model="mistral-medium-3-5")

        chain = (
            st.session_state.retriever
            | RunnableLambda(lambda docs: get_context(docs, query))
            | template
            | llm
            | parser
        )
        answer = chain.invoke(query)
        answer = clean_for_tts(answer)
        st.session_state.answer = answer

    with st.spinner("Generating spoken response..."):
        answer_audio_path = os.path.join(work_dir, "response.wav")
        text_to_speech(answer, output_path=answer_audio_path)
        st.session_state.answer_audio_path = answer_audio_path

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.transcript:
    st.markdown("#### 🗣️ Your Question")
    st.info(st.session_state.transcript)

if st.session_state.answer:
    st.markdown("#### 💬 Answer")
    st.success(st.session_state.answer)

if st.session_state.answer_audio_path and os.path.exists(st.session_state.answer_audio_path):
    st.markdown("#### 🔊 Listen to the Answer")
    st.audio(st.session_state.answer_audio_path)

st.markdown(
    '<div class="footer-note">Built with Streamlit · LangChain · FAISS · Mistral · Whisper · Piper TTS</div>'
    '<div class="footer-note">Built by Gaurav Gupta &nbsp;·&nbsp; '
    '<a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank" '
    'style="color:#93c5fd; text-decoration:none;">Connect on LinkedIn</a></div>',
    unsafe_allow_html=True,
)
