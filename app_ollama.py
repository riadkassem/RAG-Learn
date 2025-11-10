# app_streamlit_ollama_rag_clean.py
import os
import subprocess
import traceback

import torch
import whisper
import streamlit as st
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------
# Directories
# -----------------------------
UPLOADS_DIR = "data/uploads"
AUDIO_DIR = "data/audio"
TRANSCRIPTS_DIR = "data/transcripts"
FAISS_INDEX_DIR = "faiss_index"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# -----------------------------
# Device detection
# -----------------------------
device_str = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: {device_str}")

# -----------------------------
# Helpers
# -----------------------------
def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0

def transcribe_audio(audio_files, model):
    documents = []
    for audio_path in audio_files:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{base}.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            result = model.transcribe(audio_path)
            text = result.get("text", "")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(text)
        loader = TextLoader(transcript_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)
    return documents

# -----------------------------
# Whisper model (cached)
# -----------------------------
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)

# -----------------------------
# Build or update FAISS
# -----------------------------
def build_or_update_vector_store(video_paths):
    audio_files = []
    for video_path in video_paths:
        audio_name = os.path.join(AUDIO_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        if not os.path.exists(audio_name):
            ok = extract_audio(video_path, audio_name)
            if not ok:
                st.warning(f"ffmpeg failed to extract audio for {video_path}. Skipping.")
                continue
        audio_files.append(audio_name)

    if not audio_files:
        return None, None

    whisper_model = load_whisper_model()
    documents = transcribe_audio(audio_files, whisper_model)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': embed_device},
        encode_kwargs={'normalize_embeddings': True}
    )

    if any(os.scandir(FAISS_INDEX_DIR)):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(chunks)
        except Exception as e:
            st.warning(f"Failed to load existing FAISS index, creating a new one. Error: {e}")
            vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store, embeddings

# -----------------------------
# Ollama helper
# -----------------------------
def ask_ollama(prompt, model_name="gpt-oss:120b-cloud"):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.get("message", {}).get("content", "I don't know")
    except Exception as e:
        st.error(f"Ollama query failed: {e}")
        return "I don't know"

# -----------------------------
# Clean answer
# -----------------------------
def clean_answer(text: str) -> str:
    if not text:
        return ""
    for end in [".", "?", "!"]:
        idx = text.rfind(end)
        if idx != -1:
            return text[:idx+1].strip()
    return text.strip()

# -----------------------------
# Streamlit app
# -----------------------------
st.title("RAG-Learn: Lecture Video Chatbot (Ollama)")
st.write("Upload your lecture videos and ask questions!")

uploaded_videos = st.file_uploader(
    "Drag & drop lecture videos here",
    type=[ext.replace('.', '') for ext in VIDEO_EXTENSIONS],
    accept_multiple_files=True
)

question = st.text_input("Enter your question here:")

# Load FAISS index if exists
vector_store = None
if any(os.scandir(FAISS_INDEX_DIR)):
    try:
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': embed_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load existing FAISS index: {e}")

if st.button("Ask") and question:
    video_paths = []
    if uploaded_videos:
        for v in uploaded_videos:
            save_path = os.path.join(UPLOADS_DIR, v.name)
            with open(save_path, "wb") as f:
                f.write(v.getbuffer())
            video_paths.append(save_path)

        with st.spinner("Processing videos (audio extraction, transcription, embeddings)..."):
            vector_store, embeddings = build_or_update_vector_store(video_paths)
            if vector_store is None:
                st.error("No audio/transcription could be produced from the uploaded videos.")
                st.stop()

    if vector_store is None:
        st.warning("Please upload at least one lecture video or ensure FAISS index exists.")
        st.stop()

    # -----------------------------
    # FAISS similarity search
    # -----------------------------
    context_docs = vector_store.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # -----------------------------
    # Build prompt
    # -----------------------------
    prompt_template = f"""Use the following context to answer the question at the end. Follow these rules:
1. If you don't know the answer, just say "I don't know".
2. Answer concisely, maximum 5 sentences.

Context:
{context_text}

Question: {question}

Helpful Answer:"""

    # -----------------------------
    # Ask Ollama
    # -----------------------------
    with st.spinner("Running retrieval + generation via Ollama..."):
        answer = ask_ollama(prompt_template)
        answer = clean_answer(answer)

    if not answer.strip() or answer.strip().lower() == "i don't know":
        st.info("I don't know.")
    else:
        st.subheader("Answer:")
        st.write(answer)

st.write("---")
st.write("Notes:")
st.write("- Uploads are saved to `data/uploads/` and audio extracted to `data/audio/`.")
st.write("- FAISS index is stored in `faiss_index/`.")
st.write("- Ollama gpt-oss:120b-cloud is used for all generation. You must be signed in via `ollama signin`.")
st.write("- GPU is automatically used if available for transcription and embeddings.")
