"""
Name Parser: LLM Module

This module uses the Groq API (Llama 3) to parse names into
first name, last name, and title.
"""

import os
from pathlib import Path

import yaml

# Load .env file (if present)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from groq import Groq


def load_config() -> dict:
    """Loads configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_name(name: str, config: dict = None) -> dict:
    """
    Parses a name into first name, last name, and title.

    Args:
        name: The name to analyze
        config: Configuration dict (loads from file if None)

    Returns:
        Dictionary with first_name, last_name, title
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. "
            "Please edit .env and add your API key. "
            "Get your key at: https://console.groq.com/keys"
        )

    # Create client
    client = Groq(api_key=api_key)

    # Get settings from config
    llm_config = config.get("llm", {})
    model = llm_config.get("model", "llama-3.1-8b-instant")
    temperature = llm_config.get("temperature", 0.1)
    max_tokens = llm_config.get("max_tokens", 200)
    system_prompt = config.get("prompt", "Parse the name into first_name, last_name, and title.")

    # Send request
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this name: {name}"}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return _parse_response(response.choices[0].message.content)


def _parse_response(text: str) -> dict:
    """Parses the LLM response into a dictionary."""
    result = {"first_name": "", "last_name": "", "title": ""}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("first_name:"):
            result["first_name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("last_name:"):
            result["last_name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            result["title"] = "" if title.lower() == "none" else title

    return result
