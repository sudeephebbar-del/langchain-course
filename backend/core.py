"""Chroma RAG + OpenAI chat for the LangChain documentation helper."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_DIR = os.path.join(_ROOT, "chroma_db")

_prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _embeddings, _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    _embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=False,
    )
    _vectorstore = Chroma(
        persist_directory=_CHROMA_DIR,
        embedding_function=_embeddings,
    )
    return _vectorstore


def run_llm(question: str) -> Dict[str, Any]:
    """Retrieve from local Chroma DB and return an answer plus context docs for sources."""
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs: List[Document] = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model_name=model, temperature=0)
    messages = _prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return {"answer": response.content, "context": docs}
