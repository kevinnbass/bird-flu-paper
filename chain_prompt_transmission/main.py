#!/usr/bin/env python3
"""
main.py
High-level script that orchestrates article processing using the updated multi-phase pipeline.
(Extract → Contextualize → Trim → Merge → Temporal → Remainder → Mechanism → Validate)
"""

import os
import sys
import json
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv
from tqdm import tqdm
import yaml

# Local imports
from src.api import APIClient
from src.processor import ArticleProcessor, PhaseTracker
from src.validators import InputValidator, ArticleSchema


class ConfigDict(TypedDict):
    files: Dict[str, Any]
    logging: Dict[str, str]
    processing: Dict[str, Dict[str, int]]
    api: Dict[str, Any]


processor = None  # Global for signal handler access


def load_config(config_path: Path) -> ConfigDict:
    """
    Load YAML config from the given path.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def signal_handler(signum, frame):
    """
    Handle signals (e.g., SIGINT or SIGTERM).
    If 'processor' has a phase_tracker, call its cleanup before exiting.
    """
    if processor and processor.phase_tracker:
        processor.phase_tracker.cleanup()
    sys.exit(1)


# Register signal handlers for graceful termination
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def setup_logging(config: ConfigDict) -> None:
    """
    Configure logging settings based on the config file.
    """
    log_path = Path(config["logging"]["filename"]).parent
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=config["logging"]["filename"],
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"]
    )


def load_existing_output(filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load existing JSONL output to avoid reprocessing the same articles.
    Returns (list_of_records, lookup_dict_by_id).
    """
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError:
                        logging.warning("Skipping invalid JSON line in output file.")
        logging.info(f"Loaded {len(data)} existing records from {filename}")
    except FileNotFoundError:
        logging.info(f"No existing output file found at {filename}")

    lookup_dict = {a["id"]: a for a in data if "id" in a}
    return data, lookup_dict


def get_next_output_file(base_name: str) -> str:
    """
    Given a base name (like 'output'), generate 'output_1.jsonl', 'output_2.jsonl', etc.
    until we find a file that doesn't exist yet.
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.jsonl"
        if not Path(filename).exists():
            return filename
        counter += 1


def append_jsonl(data: Dict[str, Any], filename: str) -> None:
    """
    Append a single record to a JSONL file.
    """
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_articles_no_range(input_file: str) -> List[Dict[str, Any]]:
    """
    Read articles from JSONL input with no ID-range filtering.
    Returns the articles in the order they appear in the file.
    """
    articles = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    article = json.loads(line)
                    articles.append(article)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line in input file")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        raise

    if not articles:
        raise ValueError(f"No valid articles found in {input_file}")

    return articles


def handle_output_file(output_filename: str, config: ConfigDict) -> Tuple[str, Dict[str, Any]]:
    """
    If the desired output file exists, ask user whether to continue or create a new file.
    Return (filename_to_use, existing_dict).
    """
    if Path(output_filename).exists():
        print(f"Found existing {output_filename}.")
        choice = input("Would you like to continue adding to this file? [yes/no] ").strip().lower()

        if choice.startswith('y'):
            _, existing_dict = load_existing_output(output_filename)
            return output_filename, existing_dict

        choice2 = input("Create a new output file? [yes/no] ").strip().lower()
        if choice2.startswith('y'):
            base_name = config["files"]["output"]["base_name"]
            output_filename = get_next_output_file(base_name)
            print(f"Using new output file: {output_filename}")
            return output_filename, {}

        print("Terminating script at user request.")
        sys.exit(0)

    print(f"No existing {output_filename}, starting fresh.")
    return output_filename, {}


def main():
    """
    Main entry point for the script. 
    Orchestrates:
      1. Loading config
      2. Setting up logging
      3. Initializing the ArticleProcessor
      4. Handling output files
      5. Reading & reversing articles
      6. Running the multi-phase pipeline
      7. Writing final output (and any final discards)
    """
    global processor

    # 1) Load .env and read API key
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    # 2) Set up config paths
    config_path = Path("config/config.yaml")
    prompts_path = Path("config/prompts.yaml")

    # 3) Load config
    try:
        config = load_config(config_path)
        if not prompts_path.exists():
            raise ValueError(f"Prompts file not found: {prompts_path}")
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # 4) Configure logging
    setup_logging(config)

    # 5) Initialize processor with API client, validator, and prompts
    try:
        api_client = APIClient(api_key=api_key, config_path=config_path)
        validator = InputValidator(ArticleSchema())
        processor = ArticleProcessor(
            api_client=api_client,
            validator=validator,
            config_path=config_path,
            prompts_path=prompts_path
        )
    except Exception as e:
        logging.error(f"Error initializing components: {str(e)}")
        sys.exit(1)

    # 6) Determine output file
    output_filename, existing_dict = handle_output_file(
        config["files"]["output"]["base"],
        config
    )

    # 7) Load articles (ALL, no ID-range filter) and reverse them
    try:
        articles = load_articles_no_range(config["files"]["input"])
        articles.reverse()  # Process last article first
        logging.info(f"Loaded {len(articles)} articles for processing (in reverse order)")

        # We'll track final-phase discards in discard_list
        discard_list = []

        # 8) Process each article
        with tqdm(total=len(articles), desc="Processing Articles") as pbar:
            for article in articles:
                aid = article["id"]

                # Skip if already processed
                if aid in existing_dict:
                    logging.info(f"Article {aid} already processed. Skipping.")
                    pbar.update(1)
                    continue

                # Skip if missing fulltext
                if "fulltext" not in article:
                    logging.error(f"Article {aid} missing 'fulltext'. Skipping.")
                    pbar.update(1)
                    continue

                try:
                    # Run the entire multi-phase pipeline
                    final_data, discard_info = processor.process_article(article)

                    # If final_data is None, something failed
                    if final_data is not None and "error" not in final_data:
                        # Write final data to main output
                        append_jsonl(final_data, output_filename)
                        # If there's final-phase discard info (from validate), gather it
                        if discard_info:
                            discard_list.append(discard_info)
                    else:
                        logging.error(f"Processing failed for article {aid}")
                        article["processing_error"] = "Processing failed"
                        append_jsonl(article, output_filename)

                except RuntimeError as re:
                    # If an error is raised, mark the article as error
                    logging.error(str(re))
                    article["processing_error"] = str(re)
                    append_jsonl(article, output_filename)

                pbar.update(1)

        # 9) If there are final-phase discarded statements (from validate), write them out
        if discard_list:
            discard_filename = config["files"]["output"]["discarded"]
            with open(discard_filename, 'w', encoding='utf-8') as f:
                for discard in discard_list:
                    json.dump(discard, f, ensure_ascii=False)
                    f.write('\n')
            logging.info(f"Saved {len(discard_list)} discarded records to {discard_filename}")

    except Exception as e:
        # If anything major fails, log and exit
        logging.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        sys.exit(1)

    # 10) Done
    print(f"Processing complete. Results written to {output_filename}")
    if discard_list:
        print(f"Discarded statements written to {config['files']['output']['discarded']}")

if __name__ == "__main__":
    main()
