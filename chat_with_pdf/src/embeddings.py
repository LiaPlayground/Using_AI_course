"""
Embedding generation module using Ollama.

This module creates semantic embeddings for text using
local Ollama models, enabling similarity search.
"""

import logging

import ollama

logger = logging.getLogger(__name__)


def create_embedding(
    text: str,
    model: str = "nomic-embed-text"
) -> list[float]:
    """
    Create an embedding vector for text using Ollama.

    Args:
        text: The text to embed
        model: Ollama embedding model to use

    Returns:
        List of floats representing the embedding vector
    """
    logger.debug(f"Creating embedding for {len(text)} characters")

    response = ollama.embeddings(
        model=model,
        prompt=text
    )

    embedding = response['embedding']
    logger.debug(f"Created embedding with {len(embedding)} dimensions")

    return embedding


def embed_chunks(
    chunks: list[dict],
    model: str = "nomic-embed-text"
) -> list[dict]:
    """
    Create embeddings for a list of text chunks.

    Args:
        chunks: List of chunk dictionaries with 'text' key
        model: Ollama embedding model to use

    Returns:
        List of chunks with added 'embedding' key
    """
    logger.info(f"Embedding {len(chunks)} chunks using model: {model}")

    embedded_chunks = []

    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk['text'], model)

        embedded_chunk = chunk.copy()
        embedded_chunk['embedding'] = embedding
        embedded_chunks.append(embedded_chunk)

        if (i + 1) % 10 == 0:
            logger.info(f"Embedded {i + 1}/{len(chunks)} chunks")

    logger.info(f"Completed embedding {len(embedded_chunks)} chunks")
    return embedded_chunks
