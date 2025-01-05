import json
import openai
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai.error import OpenAIError
import logging
import re
import math

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Configure logging
logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify that the API key was loaded successfully
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# File paths
MERGE_SOURCE_FILE = 'final_for_analysis_2.json'  # Source file to merge from
TARGET_FILE = 'processed_all_articles_with_quotes.json'  # Target file to merge into and process

# OpenAI API parameters
MODEL = 'gpt-4o'  # Ensure this is the correct model name
MAX_RETRIES = 2
SLEEP_TIME = 5  # Base wait time in seconds for retries

# Cost parameters (unchanged as they might be useful for tracking)
INPUT_TOKEN_COST_PER_M = 1.25   # $1.25 per 1M input tokens
OUTPUT_TOKEN_COST_PER_M = 5.00  # $5.00 per 1M output tokens

# Initialize total tokens counters
total_input_tokens = 0
total_output_tokens = 0

# ------------------------------ #
#          Prompts Definitions    #
# ------------------------------ #

# Prompt for extracting quotations and interviewees with discrete fields
PROMPT_EXTRACT_QUOTES = """
You are an analytical assistant. For the following article text, perform the following tasks:

1. Identify all instances where an expert or interviewee is quoted.
2. Extract the exact quotation and the name of the expert or interviewee.

**Instructions:**
- Do not include any Markdown, code fences, or additional text.
- Respond only with a JSON object containing discrete fields for each quotation and interviewee, numbered sequentially.
- Only include fields for actual quotations and interviewees. Do not include empty fields.
- The numbering should start at 1 and increment by 1 for each quotation and interviewee pair.
- Example format:
{{
  "quotation_1": "First quotation.",
  "interviewee_1": "First interviewee.",
  "quotation_2": "Second quotation.",
  "interviewee_2": "Second interviewee."
}}

**Article Text:**
{fulltext}
"""

# ------------------------------ #
#          Helper Function        #
# ------------------------------ #

def parse_valid_json(raw_response):
    """
    Attempts to parse the raw API response into a valid JSON object.
    If the response is incomplete, it extracts only the complete quotation and interviewee pairs.

    Args:
        raw_response (str): The raw JSON response from the API.

    Returns:
        dict: Parsed JSON object with complete quotation and interviewee pairs.
    """
    # Regular expression to match quotation_n and interviewee_n pairs
    pattern = r'"quotation_(\d+)":\s*"([^"]+)",\s*"interviewee_\1":\s*"([^"]+)"'

    matches = re.findall(pattern, raw_response)
    result = {}
    for match in matches:
        index, quotation, interviewee = match
        result[f'quotation_{index}'] = quotation
        result[f'interviewee_{index}'] = interviewee

    return result

def make_api_call(prompt_template, **kwargs):
    """
    Makes an API call with the given prompt and returns the response.

    Implements exponential backoff for retries.

    Args:
        prompt_template (str): The prompt template with placeholders.
        **kwargs: The variables to format into the prompt.

    Returns:
        dict: Parsed JSON response from the API.
        int: Number of input tokens used.
        int: Number of output tokens generated.
    """
    prompt = prompt_template.format(**kwargs)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an analytical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=250,
            )
            reply = response['choices'][0]['message']['content'].strip()

            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

            global total_input_tokens, total_output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"Raw API Response:\n{reply}\n")
            logging.info(f"API Response:\n{reply}\n")
            logging.info(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")

            # Clean reply if it's enclosed in code fences
            if reply.startswith("```") and reply.endswith("```"):
                reply = '\n'.join(reply.split('\n')[1:-1]).strip()

            try:
                # Attempt to parse the JSON
                result = json.loads(reply)
                return result, input_tokens, output_tokens
            except json.JSONDecodeError:
                # If parsing fails, attempt to extract valid parts
                logging.warning("JSON parsing failed. Attempting to extract valid JSON segments.")
                result = parse_valid_json(reply)
                if result:
                    logging.info("Successfully extracted valid JSON segments.")
                    return result, input_tokens, output_tokens
                else:
                    raise ValueError("No valid JSON segments found.")

        except (OpenAIError, ValueError) as e:
            logging.error(f"Error during API call: {e}")
            if attempt < MAX_RETRIES - 1:
                # Implement exponential backoff
                backoff_time = SLEEP_TIME * math.pow(2, attempt)
                print(f"Error: {e}. Retrying in {backoff_time} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})\n")
                time.sleep(backoff_time)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Returning empty result.\n")
                logging.error(f"Failed after {MAX_RETRIES} attempts for prompt.")
                return {}, 0, 0

def extract_quotes(fulltext):
    """
    Extracts quotations and interviewees from the article text.

    Args:
        fulltext (str): The full text of the article.

    Returns:
        dict: Contains discrete 'quotation_n' and 'interviewee_n' fields.
        int: Input tokens used.
        int: Output tokens generated.
    """
    result, input_tokens, output_tokens = make_api_call(PROMPT_EXTRACT_QUOTES, fulltext=fulltext)
    return result, input_tokens, output_tokens

# ------------------------------ #
#           Main Execution        #
# ------------------------------ #

def main():
    global total_input_tokens, total_output_tokens

    # -------------------------- #
    #        Step 1: Merging      #
    # -------------------------- #

    print("Starting Step 1: Merging HuffPost Entries...\n")
    logging.info("Starting Step 1: Merging HuffPost Entries...\n")

    # Load the target JSON data
    if os.path.exists(TARGET_FILE):
        try:
            with open(TARGET_FILE, 'r', encoding='utf-8') as target_file:
                target_articles = json.load(target_file)
            print(f"Loaded existing target file '{TARGET_FILE}' with {len(target_articles)} articles.")
            logging.info(f"Loaded existing target file '{TARGET_FILE}' with {len(target_articles)} articles.")
        except json.JSONDecodeError:
            print(f"Target file '{TARGET_FILE}' is not a valid JSON. Initializing as empty list.")
            logging.error(f"Target file '{TARGET_FILE}' is not a valid JSON. Initializing as empty list.")
            target_articles = []
    else:
        print(f"Target file '{TARGET_FILE}' not found. Initializing as empty list.")
        logging.info(f"Target file '{TARGET_FILE}' not found. Initializing as empty list.")
        target_articles = []

    # Load the source JSON data
    try:
        with open(MERGE_SOURCE_FILE, 'r', encoding='utf-8') as source_file:
            source_articles = json.load(source_file)
        print(f"Loaded source file '{MERGE_SOURCE_FILE}' with {len(source_articles)} articles.")
        logging.info(f"Loaded source file '{MERGE_SOURCE_FILE}' with {len(source_articles)} articles.")
    except FileNotFoundError:
        print(f"Source file '{MERGE_SOURCE_FILE}' not found.")
        logging.error(f"Source file '{MERGE_SOURCE_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Source file '{MERGE_SOURCE_FILE}' is not a valid JSON.")
        logging.error(f"Source file '{MERGE_SOURCE_FILE}' is not a valid JSON.")
        return

    # Filter HuffPost entries
    huffpost_entries = [article for article in source_articles if article.get("media_outlet") == "HuffPost"]
    print(f"Filtered {len(huffpost_entries)} HuffPost articles from the source.")
    logging.info(f"Filtered {len(huffpost_entries)} HuffPost articles from the source.")

    # Append HuffPost entries to target_articles
    target_articles.extend(huffpost_entries)
    print(f"Merged {len(huffpost_entries)} HuffPost articles into '{TARGET_FILE}'.")
    logging.info(f"Merged {len(huffpost_entries)} HuffPost articles into '{TARGET_FILE}'.")

    # Write the merged data back to the target file
    try:
        with open(TARGET_FILE, 'w', encoding='utf-8') as target_file:
            json.dump(target_articles, target_file, ensure_ascii=False, indent=4)
        print(f"Step 1 complete. Merged data saved to '{TARGET_FILE}'.\n")
        logging.info(f"Step 1 complete. Merged data saved to '{TARGET_FILE}'.\n")
    except Exception as e:
        print(f"Error writing to target file '{TARGET_FILE}': {e}")
        logging.error(f"Error writing to target file '{TARGET_FILE}': {e}")
        return

    # -------------------------- #
    #      Step 2: Extract Quotes#
    # -------------------------- #

    print("Starting Step 2: Extracting Quotations from HuffPost Articles...\n")
    logging.info("Starting Step 2: Extracting Quotations from HuffPost Articles...\n")

    # Load the merged JSON data
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as infile:
            articles = json.load(infile)
        print(f"Loaded '{TARGET_FILE}' with {len(articles)} articles for processing.")
        logging.info(f"Loaded '{TARGET_FILE}' with {len(articles)} articles for processing.")
    except FileNotFoundError:
        print(f"Target file '{TARGET_FILE}' not found.")
        logging.error(f"Target file '{TARGET_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Target file '{TARGET_FILE}' is not a valid JSON.")
        logging.error(f"Target file '{TARGET_FILE}' is not a valid JSON.")
        return

    # Identify HuffPost articles
    huffpost_articles = [article for article in articles if article.get("media_outlet") == "HuffPost"]
    num_huffpost = len(huffpost_articles)
    print(f"Total HuffPost articles to process: {num_huffpost}\n")
    logging.info(f"Total HuffPost articles to process: {num_huffpost}\n")

    if num_huffpost == 0:
        print("No HuffPost articles found for processing. Exiting.")
        logging.info("No HuffPost articles found for processing. Exiting.")
        return

    # Identify already processed articles (those with 'quotation_n' fields)
    already_processed = [article for article in huffpost_articles if any(key.startswith('quotation_') for key in article)]
    to_process = [article for article in huffpost_articles if not any(key.startswith('quotation_') for key in article)]

    num_already_processed = len(already_processed)
    num_to_process = len(to_process)

    print(f"Articles already processed (with 'quotation_n' fields): {num_already_processed}")
    print(f"Articles to be processed: {num_to_process}\n")
    logging.info(f"Articles already processed (with 'quotation_n' fields): {num_already_processed}")
    logging.info(f"Articles to be processed: {num_to_process}\n")

    if num_to_process == 0:
        print("No unprocessed HuffPost articles found for Step 2. Skipping extraction.")
        logging.info("No unprocessed HuffPost articles found for Step 2. Skipping extraction.")
    else:
        # Process each unprocessed HuffPost article with a progress bar
        for idx, article in enumerate(tqdm(to_process, desc="Step 2: Extracting Quotations")):
            fulltext = article.get('fulltext', '').strip()

            if not fulltext:
                print(f"Article {idx + 1} has empty 'fulltext'. Skipping...\n")
                logging.info(f"Article {idx + 1} has empty 'fulltext'.")
                continue

            # Extract quotations and interviewees
            extracted_data, input_tokens, output_tokens = extract_quotes(fulltext)

            # Check if any data was extracted
            if not extracted_data:
                print(f"Article {idx + 1} - No quotations or interviewees found.\n")
                logging.info(f"Article {idx + 1} - No quotations or interviewees found.\n")
                continue

            # Add quotations and interviewees to the article
            for key, value in extracted_data.items():
                article[key] = value

            # Determine the number of quotations extracted
            num_quotations = len([key for key in extracted_data if key.startswith('quotation_')])
            logging.info(f"Article {idx + 1} - Extracted {num_quotations} quotations and interviewees.\n")
            print(f"Article {idx + 1} - Extracted {num_quotations} quotations and interviewees.\n")
            print(f"Tokens Used for Article {idx + 1} - Input: {input_tokens}, Output: {output_tokens}\n")

    # Write the updated data back to the target JSON file
    try:
        with open(TARGET_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(articles, outfile, ensure_ascii=False, indent=4)
        print(f"\nStep 2 complete. Updated data saved to '{TARGET_FILE}'.")
        logging.info(f"Step 2 complete. Updated data saved to '{TARGET_FILE}'.")
    except Exception as e:
        print(f"Error writing to target file '{TARGET_FILE}': {e}")
        logging.error(f"Error writing to target file '{TARGET_FILE}': {e}")
        return

    # -------------------------- #
    #         Token Summary      #
    # -------------------------- #

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_M
    total_cost = input_cost + output_cost

    # Output total tokens and cost
    print("\n--- Token Usage Summary ---")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Cost: ${total_cost:.6f}")
    print("----------------------------\n")

    # Log token usage and cost
    logging.info("--- Token Usage Summary ---")
    logging.info(f"Total Input Tokens: {total_input_tokens}")
    logging.info(f"Total Output Tokens: {total_output_tokens}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    logging.info("----------------------------\n")

if __name__ == "__main__":
    main()
