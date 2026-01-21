"""
Chat interface module.

This module handles generating answers to user questions
based on retrieved context from the vector store.
"""

import logging

import ollama

logger = logging.getLogger(__name__)


def create_prompt(question: str, context: list[dict]) -> str:
    """
    Create a prompt combining the question with retrieved context.

    Args:
        question: The user's question
        context: List of relevant chunks from vector search

    Returns:
        Formatted prompt string
    """
    context_texts = [c['text'] for c in context]
    context_str = "\n\n---\n\n".join(context_texts)

    prompt = f"""You are a helpful research assistant. Answer the question based on the provided context from a PDF document. If the answer is not in the context, say "I couldn't find information about that in the document."

Context from the document:
{context_str}

---

Question: {question}

Answer:"""

    return prompt


def generate_answer(
    question: str,
    context: list[dict],
    model: str = "llama3.2",
    temperature: float = 0.3
) -> str:
    """
    Generate an answer to a question using the LLM.

    Args:
        question: The user's question
        context: Relevant chunks from vector search
        model: Ollama model to use
        temperature: Creativity parameter

    Returns:
        Generated answer string
    """
    logger.info(f"Generating answer for: {question[:50]}...")
    logger.debug(f"Using {len(context)} context chunks")

    prompt = create_prompt(question, context)

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        options={
            'temperature': temperature
        }
    )

    answer = response['message']['content']
    logger.info(f"Generated answer ({len(answer)} characters)")

    return answer


def chat_loop(
    vector_store,
    embedding_model: str = "nomic-embed-text",
    chat_model: str = "llama3.2",
    top_k: int = 3
):
    """
    Interactive chat loop for querying the document.

    Args:
        vector_store: VectorStore instance with embedded document
        embedding_model: Model for query embedding
        chat_model: Model for answer generation
        top_k: Number of context chunks to retrieve
    """
    from embeddings import create_embedding

    print("\n" + "="*50)
    print("Chat with PDF")
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*50 + "\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Create query embedding
            query_embedding = create_embedding(question, embedding_model)

            # Search for relevant context
            context = vector_store.search(query_embedding, top_k)

            if not context:
                print("Assistant: I couldn't find any relevant information.\n")
                continue

            # Generate answer
            answer = generate_answer(
                question,
                context,
                model=chat_model
            )

            print(f"\nAssistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"Error: {e}\n")
