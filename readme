# PDF RAG Chatbot

A conversational AI chatbot that lets you chat with any PDF document using Retrieval-Augmented Generation (RAG). Built with LangChain, Pinecone, and LLaMA 3.3 70B via Groq.

## What it does

- Upload any PDF and ask questions about its content
- Retrieves the most relevant chunks using semantic search (MMR)
- Answers using LLaMA 3.3 70B — fast inference via Groq
- Remembers conversation history across multiple turns
- Won't hallucinate — tells you if the answer isn't in the document

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (via Groq) |
| Orchestration | LangChain LCEL |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | Pinecone |
| Memory | `RunnableWithMessageHistory` |
| Retrieval | MMR (Maximal Marginal Relevance) |

## Architecture

```
PDF File
   │
   ▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
Pinecone Vector Store (upsert in batches of 100)
   │
   ▼
MMR Retriever (k=3, fetch_k=10)
   │
   ▼
LangChain LCEL Chain + RunnableWithMessageHistory
   │
   ▼
LLaMA 3.3 70B (Groq) → Answer
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/laibakhan11/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key
PineCone_API_KEY=your_pinecone_api_key
```

Get your keys:
- Groq API key → [console.groq.com](https://console.groq.com)
- Pinecone API key → [app.pinecone.io](https://app.pinecone.io)

### 4. Set up Pinecone index

Create an index named `docsembedding` in your Pinecone dashboard with:
- **Dimensions:** 384
- **Metric:** cosine

### 5. Add your PDF

Place your PDF inside the `docs/` folder and update the path in `rag.py`:
```python
loader = PyPDFLoader("docs/your_file.pdf")
```

### 6. Run
```bash
python rag.py
```

## Project Structure

```
pdf-rag-chatbot/
├── docs/               # Place your PDF files here
├── rag.py              # Main chatbot script
├── requirements.txt    # Python dependencies
├── .env                # API keys (do NOT commit this)
├── .gitignore          # Excludes .env and docs/
└── README.md
```

## Key Design Decisions

- **MMR retrieval** instead of simple similarity search — reduces redundant chunks in context
- **Chunk overlap of 50** — preserves context across chunk boundaries
- **Batch upsert (100 vectors)** — avoids Pinecone rate limits on large PDFs
- **Session-based memory** — supports multiple independent user sessions