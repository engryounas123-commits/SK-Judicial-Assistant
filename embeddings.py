"""
Embeddings Module
Manages sentence-level embeddings using sentence-transformers
and stores them in a FAISS vector index for fast similarity search.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/legal_index.faiss")
METADATA_PATH = FAISS_INDEX_PATH.replace(".faiss", "_meta.pkl")


class LegalEmbeddings:
    """
    Wrapper around sentence-transformers + FAISS for legal document similarity search.
    Falls back to a simple TF-IDF cosine similarity when dependencies are unavailable.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None  # FAISS index
        self.metadata: List[Dict] = []  # parallel list of document metadata
        self._use_fallback = False
        self._fallback_docs: List[str] = []

        self._init_model()
        logger.info(
            f"LegalEmbeddings initialised (model={model_name}, "
            f"fallback={self._use_fallback})"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence-transformers model: {self.model_name}")
        except Exception as exc:
            logger.warning(f"sentence-transformers unavailable ({exc}); using TF-IDF fallback")
            self._use_fallback = True

    def _init_faiss(self, dimension: int):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalise)
            logger.info(f"FAISS IndexFlatIP created with dim={dimension}")
        except Exception as exc:
            logger.warning(f"FAISS unavailable ({exc}); no vector index")
            self.index = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        if not texts:
            return np.zeros((0, 384))

        if self._use_fallback or self.model is None:
            return self._tfidf_encode(texts)

        try:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-normalise for cosine via dot product
                show_progress_bar=False,
                batch_size=32,
            )
            return vectors.astype(np.float32)
        except Exception as exc:
            logger.error(f"Encoding error: {exc}")
            return self._tfidf_encode(texts)

    def build_index(self, documents: List[Dict]) -> bool:
        """
        Build FAISS index from a list of document dicts.
        Each dict must have a 'content' key; other keys go into metadata.
        Returns True on success.
        """
        if not documents:
            logger.warning("build_index called with empty document list")
            return False

        texts = [d.get("content", d.get("text", "")) for d in documents]
        vectors = self.encode(texts)

        if self._use_fallback or vectors.shape[0] == 0:
            self._fallback_docs = texts
            self.metadata = documents
            return True

        dim = vectors.shape[1]
        self._init_faiss(dim)

        if self.index is None:
            self._fallback_docs = texts
            self.metadata = documents
            return False

        import faiss
        self.index.add(vectors)
        self.metadata = documents
        logger.info(f"FAISS index built: {self.index.ntotal} vectors (dim={dim})")
        return True

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """
        Search the index for documents similar to query.
        Returns list of (score, metadata_dict) sorted by descending relevance.
        """
        if not query:
            return []

        query_vec = self.encode([query])
        if query_vec.shape[0] == 0:
            return []

        # FAISS path
        if self.index is not None and self.index.ntotal > 0:
            try:
                scores, indices = self.index.search(
                    query_vec, min(top_k, self.index.ntotal)
                )
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.metadata):
                        results.append((float(score), self.metadata[idx]))
                return results
            except Exception as exc:
                logger.error(f"FAISS search error: {exc}")

        # Fallback: cosine via numpy
        return self._fallback_search(query_vec[0], top_k)

    def save(self, path: str = FAISS_INDEX_PATH):
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("Nothing to save – FAISS index is None")
            return

        try:
            import faiss
            faiss.write_index(self.index, path)
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Index saved: {path}, metadata: {METADATA_PATH}")
        except Exception as exc:
            logger.error(f"Save error: {exc}")

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(path):
            logger.info(f"No existing FAISS index at {path}")
            return False

        try:
            import faiss
            self.index = faiss.read_index(path)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
            return True
        except Exception as exc:
            logger.error(f"Load error: {exc}")
            return False

    # ------------------------------------------------------------------
    # TF-IDF fallback
    # ------------------------------------------------------------------

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        """Minimal bag-of-words encoding as a fallback for encoding."""
        vocab: Dict[str, int] = {}
        tokenised = []
        for text in texts:
            tokens = set(text.lower().split())
            tokenised.append(tokens)
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        dim = max(len(vocab), 1)
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, tokens in enumerate(tokenised):
            for tok in tokens:
                if tok in vocab:
                    matrix[i, vocab[tok]] = 1.0
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        return matrix

    def _fallback_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[Tuple[float, Dict]]:
        """Cosine similarity search via numpy when FAISS is unavailable."""
        if not self._fallback_docs:
            return []

        doc_vecs = self._tfidf_encode(self._fallback_docs)
        if doc_vecs.shape[1] != query_vec.shape[0]:
            # Dimension mismatch – re-encode query in same space
            qvec = self._tfidf_encode([" ".join(self._fallback_docs[0].split()[:5])])[0]
            if len(qvec) != len(query_vec):
                return []
            query_vec = qvec

        scores = doc_vecs @ query_vec
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            (float(scores[i]), self.metadata[i])
            for i in indices
            if i < len(self.metadata)
        ]
