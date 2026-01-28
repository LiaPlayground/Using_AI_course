# Chat with PDF

Query your PDF documents using local LLMs with Ollama.

## Overview

Chat with PDF is a Retrieval-Augmented Generation (RAG) application that lets you ask questions about PDF documents. All processing happens locally on your machine - no data is sent to external servers.

## Features

- Local PDF text extraction
- Semantic search using Ollama embeddings
- Question answering with local LLMs
- No API keys or external services required
- Save and load document indexes for faster startup

## Requirements

- Python 3.12+
- Ollama (https://ollama.ai)
- Required Ollama models:
  - `llama3.3:70b` (or another chat model)
  - `nomic-embed-text` (for embeddings)

## Installation

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS (or download from https://ollama.ai)
brew install ollama
```

### 2. Download required models

```bash
ollama pull llama3.3:70b
ollama pull nomic-embed-text
```

### 3. Install Python dependencies

```bash
# Install pipenv if needed
pip install pipenv

# Install project dependencies
pipenv install

# Activate environment
pipenv shell
```

## Usage

### Basic usage

```bash
python src/main.py --pdf path/to/document.pdf
```

### Save index for faster loading

```bash
# First time: process PDF and save index
python src/main.py --pdf document.pdf --save-index

# Next time: load the saved index
python src/main.py --load-index document.index.json
```

### Verbose output

```bash
python src/main.py --pdf document.pdf --verbose
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
llm:
  model: "llama3.2"         # Chat model
  temperature: 0.3          # Creativity

embeddings:
  model: "nomic-embed-text" # Embedding model
  chunk_size: 500           # Words per chunk
  chunk_overlap: 50         # Overlap between chunks

search:
  top_k: 3                  # Number of relevant chunks
```

## Project Structure

```
chat_with_pdf/
├── data/papers/         # Your PDF files
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py    # PDF text extraction
│   ├── embeddings.py    # Ollama embeddings
│   ├── vector_store.py  # Simple vector storage
│   ├── chat.py          # Chat interface
│   └── main.py          # CLI entry point
├── config/settings.yaml # Configuration
├── logs/                # Log files
├── Pipfile              # Dependencies
└── README.md
```

## How It Works

1. **Load PDF**: Extract text from PDF using PyMuPDF
2. **Chunk**: Split text into overlapping chunks
3. **Embed**: Create embeddings for each chunk using Ollama
4. **Store**: Keep embeddings in memory (or save to file)
5. **Query**: Embed user question, find similar chunks
6. **Answer**: Generate response using context + LLM

## Hardware Requirements

| RAM | Recommended Setup |
|-----|-------------------|
| 8 GB | llama3.2:1b + nomic-embed-text |
| 16 GB | llama3.2:3b + nomic-embed-text |
| 32+ GB | llama3.1:8b + nomic-embed-text |

GPU acceleration (NVIDIA, Apple Silicon) significantly speeds up processing.

## Comparison to Cloud Solutions

| Aspect | Chat with PDF (Local) | Cloud RAG Services |
|--------|----------------------|-------------------|
| Privacy | Data stays local | Data sent to servers |
| Cost | Free | Per-query pricing |
| Speed | Depends on hardware | Generally fast |
| Internet | Not required | Required |
| Models | Limited by RAM | Latest models available |

## License

MIT License
