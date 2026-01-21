# Name Parser

Parses names into first name, last name, and title using AI.

## Overview

The Name Parser uses the Groq API (Llama 3) to intelligently analyze names and split them into their components - even for complex cases like academic titles or noble predicates.

## Requirements

- Python 3.10+
- Groq API Key (free at https://console.groq.com/keys)

## Installation

```bash
pip install pipenv
pipenv install
pipenv shell
```

## Configure API Key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

```bash
python src/main.py "Dr. Max Mustermann"
```

Output:
```
Parsing: Dr. Max Mustermann
Model:   llama-3.1-8b-instant

First name: Max
Last name:  Mustermann
Title:      Dr.
```

### More Examples

```bash
python src/main.py "Prof. Dr. Anna Maria Schmidt"
python src/main.py "Hans Müller"
python src/main.py "von Goethe, Johann Wolfgang"
```

### Custom Configuration

```bash
python src/main.py "Dr. Max Mustermann" -c my_config.yaml
```

## Configuration

Edit `config/settings.yaml` to customize the LLM model and prompt:

```yaml
# LLM Settings
llm:
  model: "llama-3.1-8b-instant"
  temperature: 0.1
  max_tokens: 200

# System prompt for the LLM
prompt: |
  You are an expert in name analysis.
  Parse names into their components: first name, last name, and title.
  ...
```

## Project Structure

```
name_parser/
├── src/
│   ├── __init__.py
│   ├── name_parser.py   # AI-based name parsing
│   └── main.py          # CLI entry point
├── config/
│   └── settings.yaml    # Configuration (model, prompt)
├── logs/
│   └── requests.log     # Request logging (gitignored)
├── Pipfile              # Dependencies
├── .env.example         # API key template
└── README.md
```

## Logging

All requests are logged to `logs/requests.log` with:
- Timestamp
- Git commit hash
- Model used
- Input and parsed output

## License

MIT License
