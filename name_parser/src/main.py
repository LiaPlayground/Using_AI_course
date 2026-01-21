#!/usr/bin/env python3
"""
Name Parser - Main Program

Parses names into first name, last name, and title using AI.

Usage:
    python main.py "Dr. Max Mustermann"
    python main.py "Prof. Dr. Anna Schmidt"
"""

import argparse
import logging
import subprocess
from pathlib import Path

from name_parser import parse_name, load_config


def get_git_hash() -> str:
    """Returns the current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def setup_logging():
    """Configures logging to file."""
    # Create log directory
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Log file
    log_file = log_dir / "requests.log"

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ]
    )

    return logging.getLogger(__name__)


def main():
    """Main function of the name parser."""

    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Parses names into first name, last name, and title",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Dr. Max Mustermann"
  python main.py "Prof. Dr. Anna Maria Schmidt"
  python main.py "Hans Müller"
  python main.py "von Goethe, Johann Wolfgang"
        """
    )
    parser.add_argument("name", help="The name to parse")
    parser.add_argument("-c", "--config", type=Path,
                        help="Path to config file (default: config/settings.yaml)")

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = load_config()

        model = config.get("llm", {}).get("model", "llama-3.1-8b-instant")

        print(f"Parsing: {args.name}")
        print(f"Model:   {model}")
        print()

        result = parse_name(args.name, config)

        print(f"First name: {result['first_name']}")
        print(f"Last name:  {result['last_name']}")
        print(f"Title:      {result['title'] or '-'}")

        # Log request and result
        git_hash = get_git_hash()
        logger.info(
            f"[{git_hash}] "
            f"Model: {model} | "
            f"Input: \"{args.name}\" | "
            f"First: \"{result['first_name']}\" | "
            f"Last: \"{result['last_name']}\" | "
            f"Title: \"{result['title'] or '-'}\""
        )

        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error for \"{args.name}\": {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
