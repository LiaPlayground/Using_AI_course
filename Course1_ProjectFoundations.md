<!--
author:   Sebastian Zug
email:    sebastian.zug@informatik.tu-freiberg.de
version:  1.0.0
language: en
narrator: US English Female
comment:  AI in Scientific Data Analysis - Part 1/2: Good Development Practices for Research Projects

import: https://raw.githubusercontent.com/liaTemplates/ExplainGit/master/README.md
        https://raw.githubusercontent.com/liascript-templates/plantUML/master/README.md
-->

[![LiaScript](https://raw.githubusercontent.com/LiaScript/LiaScript/master/badges/course.svg)](https://liascript.github.io/course/?https://raw.githubusercontent.com/LiaPlayground/Using_AI_course/refs/heads/main/Course1_ProjectFoundations.md)

# Good Development Practices for Research Projects

<!-- data-type="none" -->
| Parameter | Information |
|-----------|-------------|
| **Course:** | AI in Scientific Data Analysis |
| **Part:** | 1/2 |
| **Duration:** | 90 minutes |
| **Target Audience:** | Master students (non-CS disciplines) |

---

## Learning Objectives

By the end of this session, you will be able to:

1. Understand why good project structure matters for reproducibility
2. Organize a research project with proper folder structure
3. Manage dependencies using pipenv
4. Handle configuration and secrets safely
5. Implement logging for debugging and reproducibility
6. Use Git for basic version control

---

## 1. Why Process Matters

                              {{0-1}}
******************************************************************************

**Personal experience**

Have you ever experienced this?

> "I changed something in my analysis script three weeks ago, and now my results are different. I can't remember what I changed!"

Or perhaps:

> "My colleague sent me their code, but I can't run it on my machine..."

Do you know similar stories?

> "..."

******************************************************************************

                              {{1-2}}
******************************************************************************

**The Reproducibility Crisis**

A 2016 Nature survey found that 70% of researchers have failed to reproduce another scientist's experiments, and 50% have failed to reproduce their own experiments!

Common problems:

- [[X]] Files named "final_v2_REALLY_FINAL.py"
- [[X]] "It works on my machine" syndrome
- [[X]] Lost track of which parameters produced which results
- [[X]] Dependencies changed and code no longer runs
- [[X]] Data sets are not documented

******************************************************************************

                              {{2-3}}
******************************************************************************

**The Solution: Software Engineering Practices**

We don't need to become professional developers, but adopting a few key practices can save us countless hours:

1. **Project Structure** - Know where everything is
2. **Dependency Management** - Reproducible environments
3. **Configuration Management** - Separate code from settings
4. **Version Control** - Track all changes

Let's learn these through a practical example!

******************************************************************************

---

## 2. Our Running Example: Name Parser

                              {{0-1}}
******************************************************************************

We'll build a simple but useful tool: **A Name Parser**

```ascii
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Name      │     │    LLM      │     │   Parsed    │
│   Input     │────▶│   (Groq)    │────▶│   Output    │
│  (CLI)      │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘

Winkler, Clemens, Prof. Dr.
Bernhard von Cotta
Humboldt; Alexander
"..."                                                                                                                                   .
```

**What it does:**

1. Takes a name as input (via command line)
2. Uses an LLM (Groq API with Llama 3) to analyze the name
3. Outputs the parsed components: first name, last name, and title

> This project is motivated by real research focussing on the extraction of metadata from learning content.

******************************************************************************

                              {{1-2}}
******************************************************************************

**Example Usage:**

```bash
python src/main.py "Prof. Dr. Anna Maria Schmidt"
```

**Output:**

```
Analysiere: Prof. Dr. Anna Maria Schmidt

First:  Anna Maria
Last :  Schmidt
Title:  Prof. Dr.
```

More examples:

```bash
python src/main.py "Hans Müller"
python src/main.py "von Goethe, Johann Wolfgang"
```

> Please note: We do not cover the details of LLM integration today. Focus is on project structure and best practices!

******************************************************************************

---

## 3. Project Structure

                              {{0-1}}
******************************************************************************

**Why Structure Matters**

A well-organized project is:

- **Findable**: You know where everything is
- **Reproducible**: Others can understand and run it
- **Maintainable**: Easy to update and extend

******************************************************************************

                              {{1-2}}
******************************************************************************

**Our Name Parser Structure:**

```
name_parser/
├── src/
│   ├── __init__.py
│   ├── name_parser.py       <- LLM-based name parsing
│   └── main.py              <- Entry point / CLI
├── config/
│   └── settings.yaml        <- Configuration (model, prompt)
├── logs/
│   └── requests.log         <- Request logging (gitignored)
├── Pipfile                  <- Dependencies
├── Pipfile.lock             <- Locked versions
├── .env                     <- API keys (gitignored!)
├── .env.example             <- Template for .env
├── .gitignore
└── README.md 
```

**Key Principles:**

| Folder/File    | Purpose              | Public?    |
| -------------- | -------------------- | ---------- |
| `src/`         | Source code          | Yes        |
| `config/`      | Configuration files  | Yes        |
| `logs/`        | Log files            | No         |
| `.env`         | Secrets (API keys)   | **Never!** |
| `.env.example` | Template for secrets | Yes        |
| `Pipfile`      | Dependencies         | Yes        |
| `Pipfile.lock` | Locked versions      | Yes        |

> **Golden Rule:** Separate what changes frequently (secrets, output, logs) from what stays stable (code, configuration).

******************************************************************************

---

## 4. Dependency Management with pipenv

                              {{0-1}}
******************************************************************************

**The Problem: "It Works on My Machine"**

Your code runs perfectly on your laptop. You send it to a colleague and...

```bash
ModuleNotFoundError: No module named 'groq'
```

They install groq, and then...

```bash
AttributeError: module 'groq' has no attribute 'Client'
```

Different versions of the same package can have completely different APIs!

******************************************************************************

                              {{1-2}}
******************************************************************************

**The Solution: pipenv**

pipenv combines:

- **Virtual environments** (isolated Python installation)
- **Dependency tracking** (Pipfile)
- **Version locking** (Pipfile.lock)
- **Automatic .env loading**

```bash
# Install pipenv (one time)
pip install pipenv

# Create environment and install dependencies
pipenv install

# Activate the environment
pipenv shell

# Add a new package
pipenv install requests

# Generate lock file (for reproducibility)
pipenv lock
```

> Pipenv is one of several tools for dependency management. Alternatives include Poetry and Conda.

******************************************************************************

                              {{2-3}}
******************************************************************************

**Pipfile Example:**

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
groq = ">=0.4.0"
python-dotenv = ">=1.0.0"
pyyaml = "*"

[dev-packages]
pytest = "*"

[requires]
python_version = "3.12"
```

**Pipfile.lock** contains the exact versions used:

```json
{
    "groq": {
        "version": "==0.4.2",
        "hashes": ["sha256:abc123..."]
    }
}
```

> **Tip:** Always commit both `Pipfile` and `Pipfile.lock` to Git!

******************************************************************************

## 5. Configuration

                              {{0-1}}
******************************************************************************

**The Problem: Configuration Mixed with Code**

When settings and prompts live directly in your code, things get messy fast:

```python
def parse_name(name: str) -> dict:
    client = Groq()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": """You are an expert in name analysis.
Your task is to parse names into their components.

Extract the following fields:
- title: Academic or professional title (Dr., Prof., etc.)
- first_name: The person's first name
- last_name: The person's last name

Return ONLY valid JSON, no explanations.

Example:
Input: "Dr. Maria Schmidt"
Output: {"title": "Dr.", "first_name": "Maria", "last_name": "Schmidt"}

Input: "Prof. Dr. Hans-Peter Müller"
Output: {"title": "Prof. Dr.", "first_name": "Hans-Peter", "last_name": "Müller"}
"""
            },
            {"role": "user", "content": name}
        ]
    )
    return json.loads(response.choices[0].message.content)
```

**Problems:**

- Hard to read: business logic buried under configuration
- Hard to change: editing prompts requires modifying code
- Hard to experiment: can't easily try different models or settings

******************************************************************************

                              {{1-2}}
******************************************************************************

**The Solution: Configuration Files with YAML**

Separate your settings from your code using configuration files:

```yaml
# config/settings.yaml

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

Load the configuration in Python:

```python
import yaml
from pathlib import Path

def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

> **Benefits:** Change model or prompt without modifying code. Easy to experiment with different settings!

******************************************************************************

                              {{2-3}}
******************************************************************************

**Custom Configuration via CLI**

Allow users to specify their own configuration file:

```bash
python src/main.py "Dr. Max Mustermann" -c settings.yaml
```

In the code:

```python
parser.add_argument("-c", "--config", type=Path,
                    help="Path to config file (default: config/settings.yaml)")

# Load custom or default config
if args.config:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = load_config()
```

> **Flexibility:** Users can easily switch configurations for different experiments. Just provide a different YAML file!

******************************************************************************

---

## 6. Secrets Management

                              {{0-1}}
******************************************************************************

**What are Environment Variables?**

Environment variables are key-value pairs stored in your operating system's environment, outside of your code. Think of them as a "secret notebook" that your programs can read, but that isn't part of your source code.

```ascii
┌─────────────────────────────────────────────────────────────┐
│                    Operating System                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Environment Variables                    │    │
│  │  ┌─────────────────┬────────────────────────────┐   │    │
│  │  │ Variable Name   │ Value                      │   │    │
│  │  ├─────────────────┼────────────────────────────┤   │    │
│  │  │ PATH            │ /usr/bin:/usr/local/bin    │   │    │
│  │  │ HOME            │ /home/username             │   │    │
│  │  │ GROQ_API_KEY    │ gsk_abc123...              │   │    │
│  │  └─────────────────┴────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↑                                 │
│           Your Python program reads these                   │
└─────────────────────────────────────────────────────────────┘                                                                      .
```

**Why use them?**

1. **Security**: Secrets stay out of your code (and Git history!)
2. **Flexibility**: Different values for development, testing, production
3. **Portability**: Same code works on different machines with different configs

You can see your current environment variables in the terminal:

```bash
# Show all environment variables
env

# Show a specific one
echo $HOME
```

******************************************************************************

                              {{1-2}}
******************************************************************************

**Secrets: The .env Pattern**

**Never put secrets in your code!**

```python
# DANGER! Never do this:
api_key = "gsk_abc123..."  # Will end up on GitHub!
```

Instead, use environment variables:

```bash
# .env (gitignored!)
GROQ_API_KEY=gsk_abc123...
```

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file
api_key = os.getenv("GROQ_API_KEY")
```

******************************************************************************

                              {{2-3}}
******************************************************************************

**The .env.example Pattern**

> We want to share the example structure of our .env file without sharing actual secrets but without "content".

Create a template that IS committed to Git:

```bash
# .env.example (committed to Git)
# Copy this file to .env and fill in your values

# Get your key at: https://console.groq.com/keys
GROQ_API_KEY=your_api_key_here
```

Your colleagues can then:

```bash
cp .env.example .env
# Edit .env with their own API key
```

******************************************************************************

---

## 7. Logging

                              {{0-1}}
******************************************************************************

**Why Logging Matters**

When working with LLMs and external APIs, logging is essential for:

- **Debugging**: Understanding what went wrong when errors occur
- **Reproducibility**: Tracking which inputs produced which outputs
- **Cost Control**: Monitoring API usage and token consumption
- **Auditing**: Keeping records of all requests for later analysis

Without logging, you're flying blind:

> "My script worked yesterday, but today it gives different results. What changed?"

******************************************************************************

                              {{1-2}}
******************************************************************************

**Python's logging Module**

Python has a built-in logging module - no external dependencies needed:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/requests.log"),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Use it in your code
logger.info(f"Parsing name: {name}")
logger.debug(f"API response: {response}")
logger.error(f"Failed to parse: {error}")
```

**Log Levels:**

| Level    | When to use                              |
| -------- | ---------------------------------------- |
| DEBUG    | Detailed information for debugging       |
| INFO     | General information about program flow   |
| WARNING  | Something unexpected, but not an error   |
| ERROR    | Something failed                         |
| CRITICAL | Program cannot continue                  |

******************************************************************************

                              {{2-3}}
******************************************************************************

**Logging in Our Name Parser**

In `src/name_parser.py`:

```python
import logging

logger = logging.getLogger(__name__)

def parse_name(name: str, config: dict) -> dict:
    logger.info(f"Parsing name: {name}")

    try:
        response = client.chat.completions.create(...)
        logger.debug(f"Raw response: {response}")

        result = json.loads(response.choices[0].message.content)
        logger.info(f"Successfully parsed: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to parse '{name}': {e}")
        raise
```

**Example Log Output:**

```
2024-01-15 14:32:01 - INFO - Parsing name: Prof. Dr. Anna Schmidt
2024-01-15 14:32:02 - INFO - Successfully parsed: {'title': 'Prof. Dr.', 'first_name': 'Anna', 'last_name': 'Schmidt'}
```

******************************************************************************

                              {{3-4}}
******************************************************************************

**Best Practices for Logging**

1. **Create the logs directory** but keep it empty in Git:

   ```bash
   mkdir logs
   touch logs/.gitkeep
   ```

2. **Add logs to .gitignore** (but not the directory itself):

   ```bash
   # In .gitignore
   logs/*.log
   ```

3. **Use appropriate log levels**:
   - `DEBUG` for development (verbose)
   - `INFO` for production (important events only)

4. **Include context** in log messages:

   ```python
   # Bad
   logger.info("Request sent")

   # Good
   logger.info(f"Request sent for name='{name}' using model='{model}'")
   ```

5. **Never log secrets**:

   ```python
   # DANGER! Never do this:
   logger.debug(f"Using API key: {api_key}")
   ```

> **Remember:** Logs are for you and your future self. Write messages that will be helpful when debugging at 2 AM!

******************************************************************************

---

## 8. Version Control with Git

                              {{0-1}}
******************************************************************************

**Why Version Control?**

Instead of:

```
my_analysis_v1.py
my_analysis_v2.py
my_analysis_v2_fixed.py
my_analysis_FINAL.py
my_analysis_FINAL_v2.py
```

Use Git to track all changes with:

- Timestamps
- Author information
- Descriptive messages
- Full history
- Easy rollback

> Git is the most widely used version control system in the world!

!?[](https://www.youtube.com/watch?v=lLoJHifWTRw)

******************************************************************************

                              {{1-2}}
******************************************************************************

**Git File States**

```text @plantUML.png
@startuml
left to right direction
hide empty description
[*] --> Untracked : Create new file
Untracked --> Staged : git add
Staged --> Committed : git commit
Committed --> Modified : Edit file
Modified --> Staged : git add
Committed --> Untracked : git rm
@enduml
```

> Let's see this in action in our repository!

******************************************************************************

                              {{2-3}}
******************************************************************************

**Basic Git Workflow**

``` text @ExplainGit.eval
git commit -m V1
git commit -m V2
git commit -m V3
```

+ View a specific version: `git checkout <commit-hash>`
+ Revert a version with a new commit: `git revert <commit-hash>`
+ Create a new branch: `git branch <branch-name>`
+ Create and switch to a new branch: `git checkout -b <branch-name>`
+ Merge branches: `git merge <branch-name>`

******************************************************************************

                              {{3-4}}
******************************************************************************

**Essential Git Commands:**

```bash
# Initialize a new repository
git init

# Check status (what changed?)
git status

# Stage files for commit
git add src/name_parser.py
git add .  # Stage all changes

# Commit with a message
git commit -m "Add name parsing feature"

# View history
git log --oneline
```



******************************************************************************

                              {{4-5}}
******************************************************************************

**Good Commit Messages**

```bash
# Bad:
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "asdf"

# Good:
git commit -m "Add name parser using Groq API"
git commit -m "Fix handling of academic titles"
git commit -m "Update to Llama 3.1 model"
```

**Format:**

- Start with a verb (Add, Fix, Update, Remove)
- Be specific but concise
- Explain WHAT changed, not HOW

******************************************************************************

                              {{5-6}}
******************************************************************************

**What NOT to Commit**

Your `.gitignore` protects you, but double-check:

| Commit                    | Don't Commit         |
| ------------------------- | -------------------- |
| Source code (`*.py`)      | API keys (`.env`)    |
| `Pipfile`, `Pipfile.lock` | `__pycache__/`       |
| `config/settings.yaml`    | `logs/`              |
| `README.md`               | IDE settings         |
| `.gitignore`              | Virtual environments |

> **Before pushing:** Always run `git status` and check what's staged!

******************************************************************************

                              {{6-7}}
******************************************************************************

**The .gitignore File**

```bash
# .gitignore

# Secrets - NEVER commit!
.env

# Logs
logs/

# Python
__pycache__/
*.pyc
.venv/

# IDE
.vscode/
.idea/
```

> **Critical:** Always check your `.gitignore` before your first commit!

******************************************************************************

---

## 9. Putting It All Together

                              {{0-1}}
******************************************************************************

**Complete Project Checklist**

When starting a new research project:

- Create folder structure (`src/`, `config/`)
- Initialize pipenv: `pipenv install`
- Create `config/settings.yaml` for configuration
- Create `.env.example` template
- Set up `.gitignore` (include `.env`, `logs/`)
- Initialize Git: `git init`
- Write `README.md` with setup instructions
- Make first commit

******************************************************************************

                              {{1-2}}
******************************************************************************

**README.md Template**

````markdown
# Name Parser

Parses names into first name, last name, and title using AI.

## Requirements

- Python 3.10+
- Groq API key

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API key
3. Install dependencies: 
   ```bash
   pipenv install
   pipenv shell
   ```

## Usage

```bash
python src/main.py "Prof. Dr. Anna Maria Schmidt"
```
````

******************************************************************************

---

## Summary

**What We Learned Today:**

1. **Project Structure**: Organize files logically, keep it simple
2. **pipenv**: Reproducible Python environments
3. **Secrets**: `.env` files, NEVER commit API keys
4. **Logging**: Track API calls and debug issues effectively
5. **Git**: Track changes, write good commit messages

**Next Session:**

In Part 2, we'll build a more complex project using **local LLMs** with Ollama - no API keys needed! We'll also briefly explore the **Model Context Protocol (MCP)**.

---

## Resources

- [pipenv Documentation](https://pipenv.pypa.io/)
- [Git Tutorial (Software Carpentry)](https://swcarpentry.github.io/git-novice/)
- [The Turing Way - Reproducible Research](https://the-turing-way.netlify.app/)
- [Example Project: name_parser](./name_parser/)
