#!/usr/bin/env python3
"""
Chat with PDF - Main entry point.

Query your PDF documents using local LLMs with Ollama.

Usage:
    python main.py --pdf path/to/document.pdf
    python main.py --pdf document.pdf --save-index
    python main.py --load-index document_index.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from pdf_loader import load_and_chunk_pdf
from embeddings import embed_chunks
from vector_store import VectorStore
from chat import chat_loop


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Configure logging for the application.

    Args:
        level: Logging level
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}, using defaults")
        return {
            "llm": {"model": "llama3.3:70b", "temperature": 0.3},
            "embeddings": {"model": "nomic-embed-text", "chunk_size": 500, "chunk_overlap": 50},
            "search": {"top_k": 3},
            "logging": {"level": "INFO", "file": "logs/chat.log"}
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Chat with your PDF documents using local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pdf", "-p",
        type=Path,
        help="Path to the PDF file to chat with"
    )
    parser.add_argument(
        "--load-index", "-l",
        type=Path,
        help="Load a previously saved index instead of processing PDF"
    )
    parser.add_argument(
        "--save-index", "-s",
        action="store_true",
        help="Save the index for faster loading next time"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    setup_logging(log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("Chat with PDF starting...")

    # Validate arguments
    if not args.pdf and not args.load_index:
        parser.error("Either --pdf or --load-index is required")

    # Get settings
    emb_config = config.get("embeddings", {})
    llm_config = config.get("llm", {})
    search_config = config.get("search", {})

    try:
        if args.load_index:
            # Load existing index
            logger.info(f"Loading index from: {args.load_index}")
            vector_store = VectorStore.load(args.load_index)

        else:
            # Process PDF
            logger.info(f"Processing PDF: {args.pdf}")

            # Load and chunk
            chunks = load_and_chunk_pdf(
                args.pdf,
                chunk_size=emb_config.get("chunk_size", 500),
                overlap=emb_config.get("chunk_overlap", 50)
            )

            # Create embeddings
            print(f"Creating embeddings for {len(chunks)} chunks...")
            print("This may take a few minutes depending on your hardware...")

            embedded_chunks = embed_chunks(
                chunks,
                model=emb_config.get("model", "nomic-embed-text")
            )

            # Create vector store
            vector_store = VectorStore()
            vector_store.add_chunks(embedded_chunks)

            # Optionally save index
            if args.save_index:
                index_path = args.pdf.with_suffix('.index.json')
                vector_store.save(index_path)
                print(f"Index saved to: {index_path}")

        # Start chat
        chat_loop(
            vector_store,
            embedding_model=emb_config.get("model", "nomic-embed-text"),
            chat_model=llm_config.get("model", "llama3.2"),
            top_k=search_config.get("top_k", 3)
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
