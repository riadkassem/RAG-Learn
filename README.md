# RAG-Learn: Retrieval-Augmented Chatbot for Lecture Videos

RAG-Learn is a Python project that turns lecture videos into an interactive chatbot. It extracts audio from video lectures, transcribes them into text, builds embeddings, and allows you to ask questions about the content using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🔹 Features

- **Audio Extraction**: Automatically extracts `.wav` audio from lecture videos.  
- **Transcription**: Uses OpenAI Whisper to convert audio into text.  
- **Text Chunking & Embeddings**: Splits transcripts into manageable chunks and converts them into vector embeddings using HuggingFace transformers.  
- **Vector Database**: Stores embeddings in FAISS for fast similarity search.  
- **RAG Chatbot**: Uses a HuggingFace LLM (FLAN-T5) for answering questions based on lecture transcripts.  
- **Extensible**: Easily add new lectures or swap out models for faster or more accurate results.

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/RAG-Learn.git
cd RAG-Learn
```
2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:

```bash
pip3 install -r requirements.txt
```

## 📂 Directory Structure

```bash
RAG-Learn/
├── code.ipynb                 # Main notebook for processing and chatbot
├── app.py                     # Optional app for serving chatbot (if implemented)
├── data/
│   └── videos/                # Place your lecture video files here
├── README.md
```

## ⚡ Usage

1. Open the notebook:

```bash
jupyter notebook code.ipynb
```

2. Place your lecture video files (.mp4, .mkv, .mov, .avi) in data/videos.

3. Follow the steps in the notebook:

    - Extract audio from videos.

    - Transcribe audio to text using Whisper.

    - Chunk and embed transcripts.

    - Build or load the FAISS vector store.

    - Query the RAG chatbot with questions about your lectures.

4. Example query:

    query = "What is RAG?"
    result = retrievalQA.invoke({'query': query})
    print(result['result'])  # LLM answer

## 💡 Notes / Tips

    - For large video collections, consider using Whisper medium or large for better accuracy.

    - FAISS is simple and fast for local setups; avoids Chroma persistence issues.

    - Ensure the directory data/videos exists and contains your lecture files before running the notebook.

## 📈 Future Improvements

    - Incremental vector updates (adding new lectures without rebuilding the FAISS store).

    - Metadata filtering (e.g., filter by lecture topics).
