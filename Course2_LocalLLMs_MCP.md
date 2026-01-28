<!--
author:   Sebastian Zug
email:    sebastian.zug@informatik.tu-freiberg.de
version:  1.0.0
language: en
narrator: US English Female
comment:  AI in Scientific Data Analysis - Part 2/2: Local LLMs and MCP

import: https://raw.githubusercontent.com/liascript-templates/plantUML/master/README.md
-->

[![LiaScript](https://raw.githubusercontent.com/LiaScript/LiaScript/master/badges/course.svg)](https://liascript.github.io/course/?https://raw.githubusercontent.com/TUBAF-IfI-LiaScript/ai_course/main/Course2_LocalLLMs_MCP.md)

# Local LLMs and MCP for Research

<!-- data-type="none" -->
| Parameter | Information |
|-----------|-------------|
| **Course:** | AI in Scientific Data Analysis |
| **Part:** | 2/2 |
| **Duration:** | 90 minutes |
| **Target Audience:** | Master students (non-CS disciplines) |

---

## Learning Objectives

By the end of this session, you will be able to:

1. Understand the difference between local and cloud-based LLMs
2. Install and configure Ollama for local LLM usage
3. Build a practical "Chat with PDF" application
4. Understand the basics of Model Context Protocol (MCP)

---

## 1. Local vs. Web-Based LLMs

                              {{0-1}}
******************************************************************************

**Recap: Course 1 Example**

In Part 1, we built a Code Explainer that used the OpenAI API:

```python
# Required an API key in .env
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
```

This approach requires:

- Internet connection
- API key (costs money)
- Sending your code to external servers

**What if you're working with sensitive data?**

******************************************************************************

                              {{1-2}}
******************************************************************************

**The Local Alternative**

With local LLMs, the model runs entirely on your computer:

```ascii
┌─────────────────────────────────────────────────────────────┐
│                     Your Computer                           │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │   Your      │     │   Local     │     │   Output    │    │
│  │   Data      │────▶│   LLM       │────▶│   Result    │    │
│  │             │     │  (Ollama)   │     │             │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│                                                             │
│                  Nothing leaves your machine!               │
└─────────────────────────────────────────────────────────────┘                                                                       .
```

******************************************************************************

                              {{2-3}}
******************************************************************************

**Comparison: Cloud vs. Local**

<!-- data-type="none" -->
| Aspect          | Cloud LLMs                            | Local LLMs                 |
| --------------- | ------------------------------------- | -------------------------- |
| **Privacy**     | Data sent to external servers         | Data stays on your machine |
| **Cost**        | Per-token pricing, subscriptions      | Free (hardware cost only)  |
| **Performance** | Fast, powerful models (GPT-4, Claude) | Limited by your hardware   |
| **Internet**    | Required                              | Not required               |
| **Control**     | Vendor policies apply                 | Full control               |
| **Model Size**  | Largest available                     | Fits in your RAM           |

******************************************************************************

                              {{3-4}}
******************************************************************************

**Quiz: When to Use Local LLMs?**

Which scenarios are good candidates for local LLMs?

- [[X]] Analyzing patient medical records
- [[X]] Processing confidential company data
- [[ ]] Generating marketing copy for a public website
- [[X]] Working in the field without internet
- [[X]] Institutional compliance requires data to stay local
- [[ ]] Need the absolute best performance for a public demo

******************************************************************************

---

## 2. Ollama: Local LLMs Made Easy

                              {{0-1}}
******************************************************************************

**What is Ollama?**

Ollama is a tool that makes running local LLMs simple: It wraps different LLM models behind a unified API that runs as a local server on your machine.

- One-command installation
- Easy model downloading
- Simple API (similar to OpenAI)
- Runs on Windows, Mac, and Linux

**Website:** https://ollama.ai

!?[Ollama Tutorial](https://www.youtube.com/watch?v=UtSSMs6ObqY)

> **Note:** Ollama is not the only option for running local LLMs. Other popular projects include [Hugging Face Transformers](https://huggingface.co/), [LM Studio](https://lmstudio.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), and [vLLM](https://github.com/vllm-project/vllm).

******************************************************************************

                              {{1-2}}
******************************************************************************

**Installation**

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**

```bash
# Download from https://ollama.ai/download
# Or use Homebrew:
brew install ollama
```

**Windows:**

Download the installer from https://ollama.ai/download

******************************************************************************

                              {{2-3}}
******************************************************************************

**Basic Usage**

```bash
# Start Ollama server (runs in background)
ollama serve

# Download and run a model
ollama run llama3.2

# List available models
ollama list

# Pull a specific model
ollama pull mistral
ollama pull nomic-embed-text  # For embeddings
```

**Interactive Chat:**

```
>>> Hello, who are you?
I'm Llama, an AI assistant created by Meta. How can I help you today?

>>> Explain photosynthesis briefly.
Photosynthesis is the process plants use to convert sunlight, water,
and CO2 into glucose and oxygen...
```

******************************************************************************

                              {{3-4}}
******************************************************************************

**Model Selection Based on Hardware**

<!-- data-type="none" -->
| RAM        | Recommended Models         | Use Case                      |
| ---------- | -------------------------- | ----------------------------- |
| **8 GB**   | `llama3.2:1b`, `phi3:mini` | Simple tasks, quick responses |
| **16 GB**  | `llama3.2:3b`, `mistral`   | General purpose, good quality |
| **32 GB**  | `llama3.1:8b`, `codellama` | Complex tasks, coding         |
| **64+ GB** | `llama3.1:70b`             | Research, high quality        |

**GPU Acceleration:**

- NVIDIA GPU: Automatic CUDA support
- Apple Silicon (M1/M2/M3): Excellent performance with Metal
- AMD GPU: ROCm support on Linux

******************************************************************************

                              {{4-5}}
******************************************************************************

**Python Integration**

```python
import ollama

# Simple chat
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'user', 'content': 'What is DNA?'}
    ]
)

print(response['message']['content'])
```

**No API key needed!** The model runs locally.

******************************************************************************

                              {{5-6}}
******************************************************************************

**Adding Context with System Prompts**

Use the `system` role to provide instructions or context that guides the model's behavior:

```python
from ollama import Client

# Connect to Ollama (default: localhost:11434)
client = Client(host='http://localhost:11434')

response = client.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'system',
            'content': 'You are a biology expert. Explain concepts '
                       'at an undergraduate level. Be concise.'
        },
        {
            'role': 'user',
            'content': 'What is DNA?'
        }
    ]
)

print(response['message']['content'])
```

The **system prompt** sets the context, persona, or constraints for the entire conversation.

> **Demo**: Try modifying the system prompt to see how it affects the response!

******************************************************************************

                              {{6-7}}
******************************************************************************

**Multi-Turn Conversations**

Maintain conversation history by including previous messages:

```python
import ollama

# Start a conversation
messages = [
    {'role': 'system', 'content': 'You are a helpful science tutor.'}
]

# First turn
messages.append({'role': 'user', 'content': 'What is DNA?'})
response = ollama.chat(model='llama3.2', messages=messages)
assistant_reply = response['message']['content']
messages.append({'role': 'assistant', 'content': assistant_reply})

# Second turn - model remembers the context
messages.append({'role': 'user', 'content': 'How is it replicated?'})
response = ollama.chat(model='llama3.2', messages=messages)
print(response['message']['content'])  # Knows "it" refers to DNA
```

Each message has a `role`: `system`, `user`, or `assistant`.

******************************************************************************

                              {{7-8}}
******************************************************************************

**Additional Parameters**

Control the model's behavior with optional parameters:

```python
import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Write a haiku about DNA.'}],
    options={
        'temperature': 0.7,  # Higher = more creative (0.0-1.0)
        'top_p': 0.9,        # Nucleus sampling threshold
        'num_predict': 100,  # Max tokens to generate
    }
)
```

**Parameter Explanations:**

- **`temperature`** (values: 0.0 - 1.0): Controls the randomness of token selection. At **0.0**, the model always picks the most probable token – output is deterministic and repeatable. At **1.0**, less probable tokens are chosen more frequently, leading to more creative but also more unpredictable responses. For factual tasks, use 0.1-0.3; for creative writing, 0.7-0.9.

- **`top_p`** (values: 0.1 - 1.0): Also called "Nucleus Sampling". The model only considers the most probable tokens whose cumulative probability reaches the `top_p` threshold. At **0.1**, only the highest probability tokens are included (very focused); at **1.0**, all tokens are considered (maximum diversity). Often used together with `temperature` – a typical combination is `top_p=0.9` with moderate temperature.

- **`num_predict`** (values: 100 - 4096+): Limits the maximum number of generated tokens. One token corresponds to roughly 4 characters or 0.75 words in English. At **100**, you get short responses (~75 words); at **4096**, detailed texts can be generated. Higher values increase computation time proportionally.

******************************************************************************

                              {{9-10}}
******************************************************************************

**Custom Host Connection**

By default, Ollama connects to `http://localhost:11434`. To connect to a remote server or different port, use the `Client` class:

```python
from ollama import Client

# Connect to Ollama on a different host/port
client = Client(host='http://192.168.1.100:11434')

response = client.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])
```

**Environment Variable Alternative:**

```bash
# Set before running your Python script
export OLLAMA_HOST=http://192.168.1.100:11434
```

```python
import ollama
# Now uses the OLLAMA_HOST environment variable automatically
response = ollama.chat(model='llama3.2', messages=[...])
```

**Common Scenarios:**

<!-- data-type="none" -->
| Scenario | Host Configuration |
|----------|-------------------|
| Local (default) | `http://localhost:11434` |
| Remote server | `http://server-ip:11434` |
| Docker container | `http://host.docker.internal:11434` |
| Custom port | `http://localhost:8080` |

> **Demo**: Let's try connecting to a remote Ollama server!

******************************************************************************

## 3. Practical Example: Chat with PDF

                              {{0-1}}
******************************************************************************

**The Goal**

Build a tool that lets you "chat" with a PDF document:

```ascii
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    PDF       │     │   Vector     │     │   Question   │
│  Document    │────▶│   Store      │◀────│   from User  │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                   │                    │
        │    1. Extract     │    3. Find         │
        │       text        │    relevant        │
        │                   │    chunks          │
        ▼                   ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Text       │     │  Relevant    │     │    LLM       │
│   Chunks     │────▶│  Context     │────▶│   Answer     │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        2. Create              4. Generate
         embeddings            response                                                                                        .
```

**What is RAG and why do we need it?**

Large Language Models have two fundamental limitations:

1. **Knowledge cutoff**: LLMs only know what was in their training data. They cannot answer questions about your private documents, recent research papers, or company-internal information.
2. **Hallucinations**: When LLMs don't know something, they sometimes generate plausible-sounding but incorrect answers.

**Retrieval-Augmented Generation (RAG)** solves both problems by combining:

- **Retrieval**: Search through your documents to find relevant information
- **Augmentation**: Add this information to the LLM's prompt as context
- **Generation**: Let the LLM generate an answer based on the provided context

Instead of asking "What does the paper say about X?" and hoping the LLM knows your paper, we:

1. Find the paragraphs in your paper that mention X
2. Include those paragraphs in the prompt
3. Ask the LLM to answer based on this specific context

This way, the LLM acts as a "reasoning engine" over your data rather than relying solely on its training knowledge.

This is called **Retrieval-Augmented Generation (RAG)**.

******************************************************************************

                              {{1-2}}
******************************************************************************

**Project Structure (Familiar from Course 1!)**

```ascii
chat_with_pdf/
├── data/
│   └── papers/              <- Your PDF files
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py        <- Extract text from PDFs
│   ├── embeddings.py        <- Create embeddings with Ollama
│   ├── vector_store.py      <- Simple vector storage
│   ├── chat.py              <- Chat interface
│   └── main.py              <- Entry point
├── config/
│   └── settings.yaml        <- Configuration
├── logs/
├── Pipfile
├── Pipfile.lock
├── .gitignore
└── README.md                                                                                                                        .
```

**Notice:** Same structure as Course 1, but **no `.env` needed** - everything is local!

******************************************************************************

                              {{2-3}}
******************************************************************************

**Step 1: Loading PDFs**

```python
# src/pdf_loader.py
import pymupdf  # PyMuPDF library

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into smaller chunks for processing."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

**Explanation:**

- **`load_pdf()`**: Opens a PDF file using PyMuPDF and iterates through each page to extract text. The `get_text()` method handles different PDF encodings and layouts automatically.
- **`chunk_text()`**: Splits the extracted text into smaller segments of approximately 500 words each. This is essential for RAG because:
  1. Embedding models have input length limits
  2. Smaller chunks enable more precise semantic search
  3. The LLM context window is limited, so we only want to pass the most relevant pieces

******************************************************************************

                              {{3-4}}
******************************************************************************

**Step 2: Creating Embeddings**

Embeddings convert text to numbers that capture meaning:

```python
# src/embeddings.py
import ollama

def create_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Create an embedding vector for text."""
    response = ollama.embeddings(
        model=model,
        prompt=text
    )
    return response['embedding']

def embed_chunks(chunks: list) -> list:
    """Create embeddings for all text chunks."""
    embeddings = []
    for chunk in chunks:
        embedding = create_embedding(chunk)
        embeddings.append({
            'text': chunk,
            'embedding': embedding
        })
    return embeddings
```

**Explanation:**

- **Embeddings**: High-dimensional vectors (e.g., 768 or 1536 numbers) that represent the semantic meaning of text. Texts with similar meanings have similar vectors.
- **`create_embedding()`**: Uses Ollama's local embedding model (`nomic-embed-text`) to convert a text string into a numerical vector. This runs entirely on your machine.
- **`embed_chunks()`**: Processes all text chunks and stores each one together with its embedding vector. This creates our searchable knowledge base.
- **Why `nomic-embed-text`?**: It's a lightweight, high-quality embedding model optimized for semantic search that runs efficiently on consumer hardware.

******************************************************************************

                              {{4-5}}
******************************************************************************

**Step 3: Simple Vector Store**

```python
# src/vector_store.py
import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """Calculate similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(
    query_embedding: list,
    stored_embeddings: list,
    top_k: int = 3
) -> list:
    """Find the most relevant chunks for a query."""
    similarities = []
    for item in stored_embeddings:
        sim = cosine_similarity(query_embedding, item['embedding'])
        similarities.append((sim, item['text']))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    return [text for _, text in similarities[:top_k]]
```

**Explanation:**

- **Cosine Similarity**: Measures the angle between two vectors. Values range from -1 (opposite) to 1 (identical). This metric is preferred over Euclidean distance because it focuses on direction (meaning) rather than magnitude.
- **`cosine_similarity()`**: Computes the dot product of two vectors divided by the product of their magnitudes. Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **`find_relevant_chunks()`**: Compares the user's question (as an embedding) against all stored chunk embeddings. Returns the `top_k` chunks with the highest similarity scores.
- **Why `top_k=3`?**: We retrieve multiple chunks to provide sufficient context, but not so many that we overwhelm the LLM's context window or include irrelevant information.

> **Note:** For production use with large document collections, consider using specialized vector databases like ChromaDB, FAISS, or Pinecone for faster similarity search.

******************************************************************************

                              {{5-6}}
******************************************************************************

**Step 4: Chat Interface**

```python
# src/chat.py
import ollama

def answer_question(
    question: str,
    context: list,
    model: str = "llama3.2"
) -> str:
    """Generate an answer based on context."""

    context_text = "\n\n".join(context)

    prompt = f"""Based on the following context, answer the question.
If the answer is not in the context, say so.

Context:
{context_text}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']
```

**Explanation:**

- **Prompt Engineering**: The prompt template instructs the LLM to:
  1. Base its answer only on the provided context (grounding)
  2. Admit when information is not available (reducing hallucinations)
- **`context_text`**: The retrieved chunks are joined with double newlines to clearly separate them for the LLM.
- **`ollama.chat()`**: Sends the prompt to the local Llama model. The model generates an answer using both its general knowledge and the specific context provided.
- **Why this works**: By providing relevant document excerpts directly in the prompt, we "augment" the LLM's knowledge with specific information from your PDFs - this is the "Augmented" in RAG.

> Do you remember our last weeks discussion? How to improve the prompt handling from software engineering perspective?

******************************************************************************

                              {{6-7}}
******************************************************************************

**Putting It All Together**

```python
# src/main.py (simplified)
from pdf_loader import load_pdf, chunk_text
from embeddings import embed_chunks, create_embedding
from vector_store import find_relevant_chunks
from chat import answer_question

# 1. Load and process PDF
text = load_pdf("data/papers/my_paper.pdf")
chunks = chunk_text(text)
embedded_chunks = embed_chunks(chunks)

# 2. Interactive chat loop
while True:
    question = input("\nYour question: ")
    if question.lower() == 'quit':
        break

    # Find relevant context
    query_emb = create_embedding(question)
    context = find_relevant_chunks(query_emb, embedded_chunks)

    # Generate answer
    answer = answer_question(question, context)
    print(f"\nAnswer: {answer}")
```

**Explanation:**

This brings together all RAG components in a complete workflow:

1. **Indexing Phase** (runs once at startup):
   - `load_pdf()`: Extract raw text from the PDF document
   - `chunk_text()`: Split into manageable pieces (500 words each)
   - `embed_chunks()`: Convert each chunk to a vector embedding and store it

2. **Query Phase** (runs for each question):
   - `create_embedding(question)`: Convert the user's question to the same vector space
   - `find_relevant_chunks()`: Find the 3 most similar chunks using cosine similarity
   - `answer_question()`: Send the question + relevant context to the LLM for answer generation

**The RAG Pipeline in Summary:**

```ascii
User Question → Embed → Search → Retrieve Top-K → Augment Prompt → LLM → Answer
```

******************************************************************************

                              {{7-8}}
******************************************************************************

**Configuration (settings.yaml)**

```yaml
# config/settings.yaml
llm:
  model: "llama3.2"
  temperature: 0.3

embeddings:
  model: "nomic-embed-text"
  chunk_size: 500

search:
  top_k: 3

logging:
  level: "INFO"
  file: "logs/chat.log"
```

**No API keys anywhere!** Compare this to Course 1's `.env` requirement.

******************************************************************************

---

## 4. Model Context Protocol (MCP)

                              {{0-1}}
******************************************************************************

**The Problem: Tool Integration**

Every AI application needs to connect to external tools:

- File systems
- Databases
- APIs
- Code execution

Currently, each integration is custom-built:

```ascii
┌─────────┐     Custom     ┌─────────┐
│   App   │───────────────▶│  Tool   │
│    A    │  Integration A │    1    │
└─────────┘                └─────────┘

┌─────────┐     Custom     ┌─────────┐
│   App   │───────────────▶│  Tool   │
│    B    │  Integration B │    1    │
└─────────┘                └─────────┘                                              .
```

This means duplicated effort for every app + tool combination!

******************************************************************************

                              {{1-2}}
******************************************************************************

**MCP: A Standard Interface**

The Model Context Protocol (MCP) is like USB for AI tools:

```ascii
                    MCP Standard
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
     ▼                  ▼                  ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│  MCP    │       │  MCP    │       │  MCP    │
│ Server  │       │ Server  │       │ Server  │
│ (Files) │       │  (DB)   │       │  (API)  │
└─────────┘       └─────────┘       └─────────┘

Any MCP-compatible client can use any MCP server!                                            .
```

******************************************************************************

                              {{2-3}}
******************************************************************************

**MCP Architecture**

```ascii
┌────────────────────────────────────────────────────────┐
│                    MCP Host                            │
│        (Claude Desktop, VS Code, Custom App)           │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 MCP Client                      │   │
│  │        (Manages connections to servers)         │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────┘
                         │ MCP Protocol
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   MCP    │  │   MCP    │  │   MCP    │
    │  Server  │  │  Server  │  │  Server  │
    │  (Files) │  │   (DB)   │  │  (Git)   │
    └──────────┘  └──────────┘  └──────────┘                                           .
```

**Key Concepts:**

- **Host**: The AI application (e.g., Claude Desktop)
- **Client**: Manages server connections
- **Servers**: Provide tools and resources

******************************************************************************

                              {{3-4}}
******************************************************************************

**When to Use MCP?**

<!-- data-type="none" -->
| Use MCP When...                | Skip MCP When...         |
| ------------------------------ | ------------------------ |
| Building reusable integrations | One-off prototype        |
| Team needs shared tools        | Single developer project |
| Multiple AI clients            | Only one client          |
| Need governance/audit          | Simple scripts           |
| Production deployment          | Quick experiments        |

**For our Chat with PDF:**

MCP could expose the PDF search as a standard tool that any AI assistant (Claude, custom apps, etc.) could use!

******************************************************************************

                              {{4-5}}
******************************************************************************

**Conceptual Example: PDF Search as MCP Server**

```python
# Conceptual - what an MCP server might look like

from mcp import Server, Tool

server = Server("pdf-search")

@server.tool("search_papers")
def search_papers(query: str, top_k: int = 3) -> list:
    """
    Search indexed PDF papers for relevant content.

    Args:
        query: The search query
        top_k: Number of results to return
    """
    # Use our existing vector search
    query_emb = create_embedding(query)
    results = find_relevant_chunks(query_emb, embeddings, top_k)
    return results
```

Now any MCP client can search your papers!

******************************************************************************

                              {{5-6}}
******************************************************************************

**Further Learning**

For a complete MCP tutorial, see:

- **MCP Documentation**: https://modelcontextprotocol.io
- **Full Tutorial**: https://github.com/SebastianZug/MCP_tutorial

MCP is still evolving, but the core concept - standardized tool interfaces - is powerful for building reusable AI infrastructure.

******************************************************************************


## 5. Summary

                              {{0-1}}
******************************************************************************

**What We Learned Today**

1. **Local vs. Cloud LLMs**
   - Local = privacy, control, no API costs
   - Cloud = power, convenience, external data

2. **Ollama**
   - Easy local LLM installation and usage
   - Model selection based on hardware
   - Python integration without API keys

3. **Chat with PDF (RAG)**
   - Extract text from PDFs
   - Create embeddings (semantic search)
   - Simple vector store
   - Generate answers with context

4. **MCP**
   - Standard interface for AI tools
   - Reusable integrations
   - Future-proofs your work

******************************************************************************

                              {{1-2}}
******************************************************************************

**Connection to Course 1**

Our Chat with PDF project uses all the practices from Course 1:

<!-- data-type="none" -->
| Practice         | Course 1 (Code Explainer) | Course 2 (Chat with PDF) |
| ---------------- | ------------------------- | ------------------------ |
| Folder structure | Same pattern              | Same pattern             |
| Configuration    | settings.yaml             | settings.yaml            |
| Secrets          | `.env` for API key        | **Not needed!**          |
| Logging          | Python logging            | Python logging           |
| Dependencies     | Pipfile                   | Pipfile                  |
| Version control  | .gitignore                | .gitignore               |

**The difference:** Local LLMs remove the need for secret management!

******************************************************************************

## Resources

**Ollama:**
- Website: https://ollama.ai
- Models: https://ollama.ai/library
- Python Library: `pip install ollama`

**PDF Processing:**
- PyMuPDF: https://pymupdf.readthedocs.io

**MCP:**
- Specification: https://modelcontextprotocol.io
- Tutorial: https://github.com/SebastianZug/MCP_tutorial

**Example Projects:**
- [code_explainer](./code_explainer/) - Course 1 example
- [chat_with_pdf](./chat_with_pdf/) - Course 2 example
