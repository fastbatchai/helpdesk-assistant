"""Load the knowledge base, chunk it, and search it three ways.

Loading documents needs only the standard library. Searching needs the `rag`
extra (sentence-transformers, chromadb, rank-bm25), imported when first used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from uuid import uuid4

from . import DATA_DIR

KB_DIR = DATA_DIR / "kb"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EVERYONE = ("all",)  # What a normal user may see. HR staff would add "hr".


@dataclass
class Doc:
    id: str
    title: str
    department: str
    updated: str
    audience: str
    text: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    department: str
    updated: str
    audience: str
    text: str


def _parse_doc(path) -> Doc:
    """Split the `---` front-matter block from the Markdown body."""
    _, header, body = path.read_text(encoding="utf-8").split("---\n", 2)
    fields = dict(line.split(": ", 1) for line in header.strip().splitlines())
    return Doc(id=path.stem, text=body.strip(), **fields)


def load_kb() -> list[Doc]:
    return [_parse_doc(path) for path in sorted(KB_DIR.glob("*.md"))]


def get_doc(doc_id: str) -> Doc:
    return _parse_doc(KB_DIR / f"{doc_id}.md")


def chunk_docs(docs: list[Doc], max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    """Cut each document into overlapping windows of words."""
    chunks = []
    for doc in docs:
        words = doc.text.split()
        step = max_words - overlap
        for number, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{number}",
                    doc_id=doc.id,
                    title=doc.title,
                    department=doc.department,
                    updated=doc.updated,
                    audience=doc.audience,
                    text=" ".join(words[start : start + max_words]),
                )
            )
    return chunks


def _tokenize(text: str) -> list[str]:
    """Lowercase words, keeping hyphens so error codes like NGL-809 stay whole."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class Retriever:
    """Keyword (BM25), vector (Chroma) and hybrid (reciprocal rank fusion) search."""

    def __init__(self, chunks: list[Chunk], model_name: str = EMBEDDING_MODEL):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self._bm25_class = BM25Okapi
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.embedder = SentenceTransformer(model_name)
        # Ephemeral clients share one store in a process, so each Retriever gets
        # its own collection. A poisoned copy in 3.6 can't leak into a clean one.
        self.collection = chromadb.EphemeralClient().create_collection(
            f"helpdesk_kb_{uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"}
        )
        self.add_chunks(chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to both indexes. BM25 has no incremental update, so it rebuilds."""
        self.chunks.extend(chunks)
        self.by_id.update({chunk.id: chunk for chunk in chunks})
        self.bm25 = self._bm25_class([_tokenize(chunk.text) for chunk in self.chunks])
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=self.embed([chunk.text for chunk in chunks]).tolist(),
            metadatas=[
                {k: v for k, v in asdict(chunk).items() if k not in ("id", "text")}
                for chunk in chunks
            ],
        )

    def embed(self, texts: list[str]):
        return self.embedder.encode(texts, normalize_embeddings=True)

    def keyword_search(self, query: str, k: int = 5, audiences=EVERYONE) -> list[Chunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda pair: pair[0], reverse=True)
        hits = [chunk for score, chunk in ranked if score > 0 and chunk.audience in audiences]
        return hits[:k]

    def vector_search(self, query: str, k: int = 5, audiences=EVERYONE) -> list[Chunk]:
        result = self.collection.query(
            query_embeddings=self.embed([query]).tolist(),
            n_results=k,
            where={"audience": {"$in": list(audiences)}},
        )
        return [self.by_id[chunk_id] for chunk_id in result["ids"][0]]

    def hybrid_search(
        self, query: str, k: int = 5, audiences=EVERYONE, pool: int = 20, rrf_k: int = 60
    ) -> list[Chunk]:
        """Reciprocal rank fusion: reward chunks ranked high by either method."""
        scores: dict[str, float] = {}
        for hits in (
            self.keyword_search(query, pool, audiences),
            self.vector_search(query, pool, audiences),
        ):
            for rank, chunk in enumerate(hits, start=1):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (rrf_k + rank)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.by_id[chunk_id] for chunk_id in best]
