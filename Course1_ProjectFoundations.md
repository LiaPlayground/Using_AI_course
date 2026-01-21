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
5. Use Git for basic version control

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
- [[X]] Accidentally shared API keys on GitHub

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

**What it does:**

1. Takes a name as input (via command line)
2. Uses an LLM (Groq API with Llama 3) to analyze the name
3. Outputs the parsed components: first name, last name, and title

```ascii
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Name      │     │    LLM      │     │   Parsed    │
│   Input     │────▶│   (Groq)    │────▶│   Output    │
│  (CLI)      │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

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

Vorname:  Anna Maria
Nachname: Schmidt
Titel:    Prof. Dr.
```

More examples:

```bash
python src/main.py "Hans Müller"
python src/main.py "von Goethe, Johann Wolfgang"
```

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

```ascii
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

******************************************************************************

                              {{2-3}}
******************************************************************************

**Key Principles:**

| Folder/File | Purpose | In Git? |
|-------------|---------|---------|
| `src/` | Source code | Yes |
| `config/` | Configuration files | Yes |
| `logs/` | Log files | No |
| `.env` | Secrets (API keys) | **Never!** |
| `.env.example` | Template for secrets | Yes |
| `Pipfile` | Dependencies | Yes |
| `Pipfile.lock` | Locked versions | Yes |

> **Golden Rule:** Separate what changes frequently (secrets, output, logs) from what stays stable (code, configuration).

******************************************************************************

                              {{3-4}}
******************************************************************************

**Quiz: Where does it belong?**

You have a file containing your Groq API key. Where should it go?

- [( )] `src/constants.py`
- [( )] `config/api_key.yaml`
- [(X)] `.env`
- [( )] `README.md`
[[?]] Think about what happens if you accidentally share your code...
[[!]] API keys and other secrets should ALWAYS go in `.env` files, which are gitignored!

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

                              {{3-4}}
******************************************************************************

**Quiz: Dependency Management**

Why is `Pipfile.lock` important?

- [( )] It makes the code run faster
- [(X)] It ensures everyone uses the exact same package versions
- [( )] It's required by Python
- [( )] It stores your API keys
[[?]] Think about reproducibility across different machines and time...

******************************************************************************

---

## 5. Configuration and Secrets

                              {{0-1}}
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

                              {{1-2}}
******************************************************************************

**The .env.example Pattern**

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

                              {{2-3}}
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

                              {{3-4}}
******************************************************************************

**Configuration Files with YAML**

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

                              {{4-5}}
******************************************************************************

**Custom Configuration via CLI**

Allow users to specify their own configuration file:

```bash
python src/main.py "Dr. Max Mustermann" -c my_config.yaml
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

******************************************************************************

---

## 6. Version Control with Git

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

******************************************************************************

                              {{1-2}}
******************************************************************************

**Git File States**

```text @plantUML.png
@startuml
hide empty description
[*] --> Untracked : Create new file
Untracked --> Staged : git add
Staged --> Committed : git commit
Committed --> Modified : Edit file
Modified --> Staged : git add
Committed --> Untracked : git rm
@enduml
```

******************************************************************************

                              {{2-3}}
******************************************************************************

**Basic Git Workflow**

``` text @ExplainGit.eval
git commit -m "Initial project structure"
git commit -m "Add name parser module"
git commit -m "Add CLI interface"
```

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

| Commit | Don't Commit |
|--------|--------------|
| Source code (`*.py`) | API keys (`.env`) |
| `Pipfile`, `Pipfile.lock` | `__pycache__/` |
| `config/settings.yaml` | `logs/` |
| `README.md` | IDE settings |
| `.gitignore` | Virtual environments |

> **Before pushing:** Always run `git status` and check what's staged!

******************************************************************************

---

## 7. Putting It All Together

                              {{0-1}}
******************************************************************************

**Complete Project Checklist**

When starting a new research project:

- [ ] Create folder structure (`src/`, `config/`)
- [ ] Initialize pipenv: `pipenv install`
- [ ] Create `config/settings.yaml` for configuration
- [ ] Create `.env.example` template
- [ ] Set up `.gitignore` (include `.env`, `logs/`)
- [ ] Initialize Git: `git init`
- [ ] Write `README.md` with setup instructions
- [ ] Make first commit

******************************************************************************

                              {{1-2}}
******************************************************************************

**README.md Template**

```markdown
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
```

******************************************************************************

---

## Summary

**What We Learned Today:**

1. **Project Structure**: Organize files logically, keep it simple
2. **pipenv**: Reproducible Python environments
3. **Secrets**: `.env` files, NEVER commit API keys
4. **Git**: Track changes, write good commit messages

**Next Session:**

In Part 2, we'll build a more complex project using **local LLMs** with Ollama - no API keys needed! We'll also briefly explore the **Model Context Protocol (MCP)**.

---

## Resources

- [pipenv Documentation](https://pipenv.pypa.io/)
- [Git Tutorial (Software Carpentry)](https://swcarpentry.github.io/git-novice/)
- [The Turing Way - Reproducible Research](https://the-turing-way.netlify.app/)
- [Example Project: name_parser](./name_parser/)
