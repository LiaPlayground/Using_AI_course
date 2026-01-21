"""
Simple vector storage and similarity search.

This module provides a basic in-memory vector store
for storing and querying embedded text chunks.
"""

import logging
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between -1 and 1 (higher = more similar)
    """
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class VectorStore:
    """
    Simple in-memory vector store for embedded text chunks.
    """

    def __init__(self):
        """Initialize an empty vector store."""
        self.chunks = []
        logger.info("Initialized empty VectorStore")

    def add_chunks(self, chunks: list[dict]):
        """
        Add embedded chunks to the store.

        Args:
            chunks: List of chunk dicts with 'text' and 'embedding' keys
        """
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks (total: {len(self.chunks)})")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> list[dict]:
        """
        Find the most similar chunks to a query.

        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of results to return

        Returns:
            List of most relevant chunks with similarity scores
        """
        logger.debug(f"Searching for top {top_k} similar chunks")

        if not self.chunks:
            logger.warning("VectorStore is empty")
            return []

        # Calculate similarities
        similarities = []
        for chunk in self.chunks:
            sim = cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append({
                'text': chunk['text'],
                'index': chunk.get('index', -1),
                'similarity': sim
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        results = similarities[:top_k]
        logger.debug(f"Found {len(results)} results (best sim: {results[0]['similarity']:.3f})")

        return results

    def save(self, file_path: str | Path):
        """
        Save the vector store to a JSON file.

        Args:
            file_path: Path for the output file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'chunks': [
                {
                    'text': c['text'],
                    'index': c.get('index', -1),
                    'embedding': c['embedding']
                }
                for c in self.chunks
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self.chunks)} chunks to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> 'VectorStore':
        """
        Load a vector store from a JSON file.

        Args:
            file_path: Path to the saved store

        Returns:
            Loaded VectorStore instance
        """
        file_path = Path(file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)

        store = cls()
        store.chunks = data['chunks']

        logger.info(f"Loaded {len(store.chunks)} chunks from {file_path}")
        return store

    def __len__(self):
        """Return the number of chunks in the store."""
        return len(self.chunks)
