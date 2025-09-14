import os
import subprocess
from pathlib import Path
import whisper
import gradio as gr

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface.llms import HuggingFacePipeline

# -----------------------------
# Directories
# -----------------------------
AUDIO_DIR = "data/audio"
TRANSCRIPTS_DIR = "data/transcripts"
FAISS_INDEX_DIR = "faiss_index"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_audio(audio_files, model):
    documents = []
    for audio_path in audio_files:
        relative_path = os.path.basename(audio_path)
        transcript_path = os.path.join(TRANSCRIPTS_DIR, os.path.splitext(relative_path)[0] + ".txt")

        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            result = model.transcribe(audio_path)
            text = result["text"]
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(text)

        loader = TextLoader(transcript_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)
    return documents

def build_or_update_vector_store(video_paths):
    # Extract audio
    audio_files = []
    for video_path in video_paths:
        audio_name = os.path.join(AUDIO_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        if not os.path.exists(audio_name):
            extract_audio(video_path, audio_name)
        audio_files.append(audio_name)

    # Transcribe
    whisper_model = whisper.load_model("small")
    documents = transcribe_audio(audio_files, whisper_model)

    # Chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Incremental FAISS update
    if os.path.exists(FAISS_INDEX_DIR):
        vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store, embeddings

# -----------------------------
# Load LLM & Prompt
# -----------------------------
hf = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    model_kwargs={"temperature": 0.1, "max_length": 128, "do_sample": True},
    pipeline_kwargs={"max_new_tokens": 128}
)

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, just say "I don't know".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# -----------------------------
# Initialize
# -----------------------------
vector_store = None
retriever = None
retrievalQA = None

if os.path.exists(FAISS_INDEX_DIR):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': PROMPT}
    )

# -----------------------------
# Gradio function
# -----------------------------
def rag_chat(uploaded_videos, question):
    global vector_store, retriever, retrievalQA

    # Convert Gradio files to paths
    video_paths = [v.name for v in uploaded_videos] if uploaded_videos else []

    if video_paths:
        vector_store, embeddings = build_or_update_vector_store(video_paths)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        retrievalQA = RetrievalQA.from_chain_type(
            llm=hf,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': PROMPT}
        )

    if retrievalQA is None:
        return "Please upload at least one lecture video first.", ""

    # Query
    result = retrievalQA.invoke({'query': question})
    answer = result['result']
    sources_text = ""
    for i, doc in enumerate(result['source_documents']):
        sources_text += f"Source #{i+1}: {doc.metadata['source']}\n{doc.page_content}\n\n"

    return answer, sources_text

# -----------------------------
# Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=rag_chat,
    inputs=[
        gr.Files(
            label="Drag & drop lecture videos here",
            file_types=list(VIDEO_EXTENSIONS)
        ),
        gr.Textbox(label="Enter your question here:")
    ],
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
    title="RAG-Learn Chatbot",
    description="Drag & drop lecture videos and ask questions!"
)


if __name__ == "__main__":
    iface.launch(share=True)
