#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import yaml

from src.api import APIClient
from src.processor import ArticleProcessor
from src.validators import InputValidator, ArticleSchema

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing logging settings
    """
    # Ensure logs directory exists
    log_path = Path(config["logging"]["filename"]).parent
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=config["logging"]["filename"],
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"]
    )

def load_existing_output(filename: str) -> List[Dict[str, Any]]:
    """
    Load existing output file if it exists.
    
    Args:
        filename (str): Path to output file
        
    Returns:
        List[Dict[str, Any]]: List of existing processed articles
    """
    existing_data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))
        logging.info(f"Loaded {len(existing_data)} existing records from {filename}")
    except FileNotFoundError:
        logging.info(f"No existing output file found at {filename}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing existing output file: {str(e)}")
        raise
    
    return existing_data

def get_next_output_file(base_name: str) -> str:
    """
    Generate next available output filename.
    
    Args:
        base_name (str): Base name for output file
        
    Returns:
        str: New output filename
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.jsonl"
        if not Path(filename).exists():
            return filename
        counter += 1

def append_jsonl(data: Dict[str, Any], filename: str) -> None:
    """
    Append a record to a JSONL file.
    
    Args:
        data (Dict[str, Any]): Data to append
        filename (str): Target file
    """
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def load_articles(input_file: str, id_range: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Load articles from input file within specified ID range.
    
    Args:
        input_file (str): Path to input file
        id_range (Dict[str, int]): Range of IDs to include
        
    Returns:
        List[Dict[str, Any]]: List of articles to process
    """
    articles = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    article = json.loads(line)
                    try:
                        article_id = int(article.get("id", ""))
                        if id_range["min"] <= article_id <= id_range["max"]:
                            articles.append(article)
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid article ID: {article.get('id', 'unknown')}")
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line in input file")
                    
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        raise
        
    return articles

def handle_output_file(output_filename: str) -> tuple[str, Dict[str, Any]]:
    """
    Handle existing output file and determine processing mode.
    
    Args:
        output_filename (str): Desired output filename
        
    Returns:
        tuple[str, Dict[str, Any]]: Final filename and dictionary of existing records
    """
    existing_dict = {}
    
    if Path(output_filename).exists():
        print(f"Found existing {output_filename}.")
        choice = input("Would you like to continue coding in this file? [yes/no] ").strip().lower()
        
        if choice.startswith('y'):
            existing_data = load_existing_output(output_filename)
            existing_dict = {a["id"]: a for a in existing_data}
            return output_filename, existing_dict
            
        choice2 = input("Create a new output file? [yes/no] ").strip().lower()
        if choice2.startswith('y'):
            output_filename = get_next_output_file('distribution_subset_transmission')
            print(f"Using new output file: {output_filename}")
            return output_filename, existing_dict
            
        print("Terminating script at user request.")
        sys.exit(0)
        
    print(f"No existing {output_filename}, starting fresh.")
    return output_filename, existing_dict

def main():
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()
    
    # Ensure API key is set
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Load configuration
    config_path = Path("config/config.yaml")
    prompts_path = Path("config/prompts.yaml")
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config)
    
    # Initialize components
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
    
    # Handle output file
    output_filename, existing_dict = handle_output_file(config["files"]["output"]["base"])
    
    # Load and process articles
    try:
        articles = load_articles(config["files"]["input"], config["processing"]["id_range"])
        logging.info(f"Loaded {len(articles)} articles for processing")
        
        discard_list = []
        with tqdm(total=len(articles), desc="Processing Articles") as pbar:
            for article in articles:
                aid = article["id"]
                
                # Skip if already processed
                if aid in existing_dict:
                    logging.info(f"Article {aid} already processed. Skipping.")
                    pbar.update(1)
                    continue
                
                # Process article
                try:
                    final_data, discard_info = processor.process_article(article)
                    
                    if final_data:
                        append_jsonl(final_data, output_filename)
                        if discard_info:
                            discard_list.append(discard_info)
                    else:
                        logging.error(f"Processing failed for article {aid}")
                        article["processing_error"] = "Failed to process article"
                        append_jsonl(article, output_filename)
                        
                except Exception as e:
                    logging.error(f"Error processing article {aid}: {str(e)}", exc_info=True)
                    article["processing_error"] = str(e)
                    append_jsonl(article, output_filename)
                    
                pbar.update(1)
        
        # Save discarded statements
        if discard_list:
            discard_filename = config["files"]["output"]["discarded"]
            with open(discard_filename, 'w', encoding='utf-8') as f:
                for discard in discard_list:
                    json.dump(discard, f, ensure_ascii=False)
                    f.write('\n')
            logging.info(f"Saved {len(discard_list)} discarded records to {discard_filename}")
        
    except Exception as e:
        logging.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        sys.exit(1)
    
    print(f"Processing complete. Results written to {output_filename}")
    if discard_list:
        print(f"Discarded statements written to {config['files']['output']['discarded']}")

if __name__ == "__main__":
    main()
