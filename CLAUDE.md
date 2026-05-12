# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a LangChain RAG (Retrieval-Augmented Generation) course project. It demonstrates how to ingest text documents into vector stores and query them using LLMs via LangChain's LCEL (LangChain Expression Language).

## Environment Setup

Requires a `.env` file with:
- `OPENAI_API_KEY` — for embeddings and LLM
- `INDEX_NAME` — Pinecone index name (for Pinecone backend)
- `PINECONE_API_KEY` — for Pinecone vector store
- `LANGSMITH_API_KEY` — for LangSmith tracing (optional)
- `LANGCHAIN_TRACING_V2=true` / `LANGSMITH_TRACING=true` — enables LangSmith observability

## Running the Code

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Step 1: Ingest documents into Pinecone
python ingestion.py

# Step 1 (alternative): Ingest into local Chroma store
python ingestion_chroma.py

# Step 2: Run the RAG query demo
python main.py
```

## Architecture

The project follows a two-phase RAG pattern:

**Ingestion phase** (`ingestion.py` / `ingestion_chroma.py`):
`TextLoader` → `CharacterTextSplitter` (1000-char chunks) → `OpenAIEmbeddings` → vector store (Pinecone or Chroma)

- `ingestion.py` uses Pinecone (cloud-hosted); requires `INDEX_NAME` env var
- `ingestion_chroma.py` uses Chroma (local); persists to `./chroma_store/`
- Source document: `mediumblog1.txt` (a Medium article about vector databases)

**Retrieval phase** (`main.py`):
Connects to Pinecone, creates a retriever (`k=3`), and demonstrates two RAG implementations side by side:
1. **Without LCEL** — manual step-by-step: retrieve → format → prompt → LLM
2. **With LCEL** — declarative pipe chain using `RunnablePassthrough.assign` and `|` operator

The LCEL chain in `main.py` uses `itemgetter("question")` to extract the question from the input dict, passes it through the retriever, formats docs, then pipes through the prompt template and LLM.

## Key Dependencies

- `langchain-core` — LCEL primitives (`RunnablePassthrough`, `StrOutputParser`, etc.)
- `langchain-openai` — `ChatOpenAI`, `OpenAIEmbeddings`
- `langchain-pinecone` — `PineconeVectorStore`
- `langchain-chroma` — `Chroma` (local alternative)
- `langchain-community` — `TextLoader`
- `langchain-text-splitters` — `CharacterTextSplitter`
- `python-dotenv` — `.env` loading
