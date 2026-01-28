"""
PDF loading and text extraction module.

This module handles loading PDF files and extracting
their text content for further processing.
"""

import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def load_pdf(file_path: str | Path) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        RuntimeError: If PDF extraction fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logger.info(f"Loading PDF: {file_path}")

    try:
        doc = pymupdf.open(file_path)
        text = ""

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            logger.debug(f"Extracted page {page_num + 1}: {len(page_text)} characters")

        page_count = len(doc)
        doc.close()

        logger.info(f"Extracted {len(text)} characters from {page_count} pages")
        return text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise RuntimeError(f"PDF extraction failed: {e}")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: The full text to split
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of chunk dictionaries with 'text' and 'index' keys
    """
    logger.info(f"Chunking text (size={chunk_size}, overlap={overlap})")

    words = text.split()
    chunks = []
    chunk_index = 0

    i = 0
    while i < len(words):
        # Get chunk of words
        end = min(i + chunk_size, len(words))
        chunk_words = words[i:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            'text': chunk_text,
            'index': chunk_index,
            'start_word': i,
            'end_word': end
        })

        chunk_index += 1
        i += chunk_size - overlap  # Move forward with overlap

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def load_and_chunk_pdf(
    file_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Load a PDF and split it into chunks in one step.

    Args:
        file_path: Path to the PDF file
        chunk_size: Words per chunk
        overlap: Word overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    text = load_pdf(file_path)
    return chunk_text(text, chunk_size, overlap)
