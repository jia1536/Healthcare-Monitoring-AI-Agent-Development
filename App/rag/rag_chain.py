"""
RAG (Retrieval-Augmented Generation) chain over local health documents.

Uses a local TF-IDF based embedding (scikit-learn) instead of a hosted
embedding API. This keeps the RAG pipeline fully offline / API-key-free
for local documents, and swaps in cleanly for a hosted embedding model
(e.g. OpenAIEmbeddings, HuggingFaceEmbeddings) later without changing
the retriever interface.
"""

import os
import glob
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "documents",
)


class TfidfEmbeddings(Embeddings):
    """A lightweight local Embeddings implementation backed by TF-IDF.

    Fits a TfidfVectorizer over the full document corpus once, then
    exposes embed_documents / embed_query so it's a drop-in for any
    LangChain VectorStore (FAISS here) expecting the Embeddings interface.
    """

    def __init__(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self.vectorizer.fit(corpus)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.vectorizer.transform([text])
        return vector.toarray().astype(np.float32)[0].tolist()


def load_local_documents(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Load all .md/.txt files from the local documents directory."""
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.md"))) + sorted(
        glob.glob(os.path.join(docs_dir, "*.txt"))
    )
    if not paths:
        raise FileNotFoundError(f"No local documents found in {docs_dir}")

    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(
            Document(page_content=text, metadata={"source": os.path.basename(path)})
        )
    return documents


def build_vectorstore(docs_dir: str = DOCS_DIR) -> FAISS:
    """Build an in-memory FAISS vectorstore from local health documents."""
    raw_documents = load_local_documents(docs_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(raw_documents)

    corpus_texts = [chunk.page_content for chunk in chunks]
    embeddings = TfidfEmbeddings(corpus_texts)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


class HealthRAGChain:
    """Retrieval chain over local health documents.

    Usage:
        rag = HealthRAGChain()
        results = rag.retrieve("What are symptoms of high blood pressure?")
    """

    def __init__(self, docs_dir: str = DOCS_DIR, k: int = 3):
        self.k = k
        self.vectorstore = build_vectorstore(docs_dir)

    def retrieve(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    def retrieve_as_context(self, query: str) -> str:
        """Return retrieved chunks formatted as a single context string
        ready to inject into an LLM prompt."""
        docs = self.retrieve(query)
        if not docs:
            return "No relevant local health information found."

        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source: {source}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    rag = HealthRAGChain()
    test_query = "What are the symptoms of diabetes?"
    print(f"Query: {test_query}\n")
    print(rag.retrieve_as_context(test_query))
