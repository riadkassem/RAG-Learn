# app_streamlit.py
import os
import subprocess
import traceback

import torch
import whisper
import streamlit as st

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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
if torch.cuda.is_available():
    device_idx = 0
    device_str = "cuda"
else:
    device_idx = -1
    device_str = "cpu"

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

    whisper_model = load_whisper_model()  # ✅ cached whisper
    documents = transcribe_audio(audio_files, whisper_model)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': embed_device},
        encode_kwargs={'normalize_embeddings': True}
    )

    index_exists = any(os.scandir(FAISS_INDEX_DIR))
    if index_exists:
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
# Build HF text2text pipeline (cached)
# -----------------------------
@st.cache_resource
def build_hf_pipeline(model_id="google/flan-t5-base", device_idx=-1, max_new_tokens=256):
    """Build and cache HF pipeline (higher max tokens for full answers)."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Utility: Ensure answers end at a full stop
# -----------------------------
def clean_answer(text: str) -> str:
    """Trim output so it ends at the last full sentence."""
    if not text:
        return ""
    for end in [".", "?", "!"]:
        idx = text.rfind(end)
        if idx != -1:
            return text[:idx+1].strip()
    return text.strip()

# -----------------------------
# Load LLM & Prompt
# -----------------------------
with st.spinner("Loading language model (may take a while on first run)..."):
    hf = build_hf_pipeline(model_id="google/flan-t5-base", device_idx=device_idx, max_new_tokens=256)

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, just say "I don't know".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {input}

Helpful Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "input"]  # ✅ must be 'input' for create_retrieval_chain
)

# -----------------------------
# Initialize FAISS if present
# -----------------------------
vector_store = None
retriever = None
retrievalQA = None

if any(os.scandir(FAISS_INDEX_DIR)):
    try:
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceBgeEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': embed_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    except Exception as e:
        st.warning(f"Could not load existing FAISS index: {e}")

# -----------------------------
# Streamlit app
# -----------------------------
st.title("RAG-Learn: Lecture Video Chatbot")
st.write("Upload your lecture videos and ask questions!")

uploaded_videos = st.file_uploader(
    "Drag & drop lecture videos here",
    type=[ext.replace('.', '') for ext in VIDEO_EXTENSIONS],
    accept_multiple_files=True
)

question = st.text_input("Enter your question here:")

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
        st.warning("Please upload at least one lecture video first (or ensure the FAISS index is present).")
        st.stop()

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    # Create chains
    qa_chain = create_stuff_documents_chain(
        llm=hf,
        prompt=PROMPT
    )
    retrievalQA = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=qa_chain
    )

    # Run retrieval + generation
    try:
        with st.spinner("Running retrieval + generation..."):
            result = retrievalQA.invoke({"input": question})  # ✅ 'input' key matches PromptTemplate
    except Exception as e:
        st.error(f"RetrievalQA failed: {e}")
        traceback.print_exc()
        st.stop()

    # Extract answer and sources
    raw_answer = result.get('answer', '')
    answer = clean_answer(raw_answer)
    source_docs = result.get('context', [])

    if not answer.strip() or answer.strip().lower() == "i don't know":
        st.info("I don't know.")
    else:
        st.subheader("Answer:")
        st.write(answer)

        if source_docs:
            st.subheader("Sources:")
            for i, doc in enumerate(source_docs):
                src = doc.metadata.get('source', 'unknown')
                st.markdown(f"**Source #{i+1}:** {src}\n\n{doc.page_content}")
        else:
            st.info("No source documents available for this answer.")

st.write("---")
st.write("Notes:")
st.write("- The app will save uploads to `data/uploads/` and extract audio to `data/audio/`.")
st.write("- FAISS index will live in the `faiss_index/` directory.")
st.write("- If you have a GPU available, models will be placed on CUDA automatically.")
