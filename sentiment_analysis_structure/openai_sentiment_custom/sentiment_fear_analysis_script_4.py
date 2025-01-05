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

# Input and output file paths
INPUT_FILE_PASS1 = 'processed_all_articles.json'  # Input for Pass 1
OUTPUT_FILE_PASS1 = 'processed_all_articles_with_quotes.json'  # Output for Pass 1

INPUT_FILE_PASS2 = OUTPUT_FILE_PASS1  # Input for Pass 2
OUTPUT_FILE_PASS2 = 'final_processed_all_articles.json'  # Output for Pass 2

# OpenAI API parameters
MODEL = 'gpt-4o'
MAX_RETRIES = 5
SLEEP_TIME = 5  # Base wait time in seconds for retries

# Cost parameters
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

# Prompt for assessing sentiment and fear with discrete fields
PROMPT_ASSESS_SENTIMENT_FEAR = """
You are an analytical assistant. Assess the following quotation for sentiment and fear.

**Instructions:**
- Do not include any Markdown, code fences, or additional text.
- Respond only with a JSON object containing discrete fields:
  - "sentiment": "positive" | "neutral" | "negative"
  - "fear": "fearmongering" | "neutral" | "reassuring"

**Quotation:**
"{quotation}"
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

def assess_sentiment_fear(quotation):
    """
    Assesses sentiment and fear for a given quotation.

    Args:
        quotation (str): The quotation text.

    Returns:
        dict: Contains 'sentiment' and 'fear' assessments.
        int: Input tokens used.
        int: Output tokens generated.
    """
    result, input_tokens, output_tokens = make_api_call(PROMPT_ASSESS_SENTIMENT_FEAR, quotation=quotation)
    sentiment = result.get('sentiment', 'neutral')
    fear = result.get('fear', 'neutral')
    return {'sentiment': sentiment, 'fear': fear}, input_tokens, output_tokens

# ------------------------------ #
#           Main Execution        #
# ------------------------------ #

def main():
    global total_input_tokens, total_output_tokens

    # -------------------------- #
    #         Pass 1: Extract Quotes and Interviewees
    # -------------------------- #

    print("Starting Pass 1: Extracting Quotations and Interviewees...\n")
    logging.info("Starting Pass 1: Extracting Quotations and Interviewees...\n")

    # Load the input JSON data for Pass 1
    try:
        with open(INPUT_FILE_PASS1, 'r', encoding='utf-8') as infile:
            articles = json.load(infile)
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE_PASS1}' not found.")
        logging.error(f"Input file '{INPUT_FILE_PASS1}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Input file '{INPUT_FILE_PASS1}' is not a valid JSON.")
        logging.error(f"Input file '{INPUT_FILE_PASS1}' is not a valid JSON.")
        return

    # Separate articles into already processed and to be processed
    already_processed = [article for article in articles if any(key.startswith('quotation_') for key in article)]
    to_process_articles = [article for article in articles if not any(key.startswith('quotation_') for key in article)]

    num_already_processed = len(already_processed)
    num_to_process = len(to_process_articles)

    print(f"Total articles loaded: {len(articles)}")
    print(f"Articles already processed (with 'quotation_n' fields): {num_already_processed}")
    print(f"Articles to be processed: {num_to_process}\n")
    logging.info(f"Total articles loaded: {len(articles)}")
    logging.info(f"Articles already processed (with 'quotation_n' fields): {num_already_processed}")
    logging.info(f"Articles to be processed: {num_to_process}\n")

    if num_to_process == 0:
        print("No unprocessed articles found for Pass 1. Skipping to Pass 2.")
        logging.info("No unprocessed articles found for Pass 1. Skipping to Pass 2.")
    else:
        # Process each unprocessed article with a progress bar
        for idx, article in enumerate(tqdm(to_process_articles, desc="Pass 1: Extracting Quotations")):
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

    # Combine already processed and newly processed articles
    all_articles_pass1 = already_processed + to_process_articles

    # Write the augmented data to the output JSON for Pass 1
    try:
        with open(OUTPUT_FILE_PASS1, 'w', encoding='utf-8') as outfile:
            json.dump(all_articles_pass1, outfile, ensure_ascii=False, indent=4)
        print(f"\nPass 1 complete. Output saved to '{OUTPUT_FILE_PASS1}'.")
        logging.info(f"Pass 1 complete. Output saved to '{OUTPUT_FILE_PASS1}'.")
    except Exception as e:
        print(f"Error writing to output file for Pass 1: {e}")
        logging.error(f"Error writing to output file for Pass 1: {e}")
        return

    # -------------------------- #
    #         Pass 2: Assess Sentiment and Fear
    # -------------------------- #

    print("\nStarting Pass 2: Assessing Sentiment and Fear...\n")
    logging.info("\nStarting Pass 2: Assessing Sentiment and Fear...\n")

    # Load the input JSON data for Pass 2
    try:
        with open(INPUT_FILE_PASS2, 'r', encoding='utf-8') as infile:
            articles = json.load(infile)
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE_PASS2}' not found.")
        logging.error(f"Input file '{INPUT_FILE_PASS2}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Input file '{INPUT_FILE_PASS2}' is not a valid JSON.")
        logging.error(f"Input file '{INPUT_FILE_PASS2}' is not a valid JSON.")
        return

    # Separate articles into already processed and to be processed
    already_processed_pass2 = []
    to_process_articles_pass2 = []

    for article in articles:
        # Determine the number of quotations by checking keys
        quotation_keys = [key for key in article.keys() if key.startswith('quotation_')]
        num_quotations = len(quotation_keys)

        # Check if sentiment and fear fields exist for all quotations
        processed = True
        for i in range(1, num_quotations + 1):
            if not (f'sentiment_{i}' in article and f'fear_{i}' in article):
                processed = False
                break

        if processed:
            already_processed_pass2.append(article)
        else:
            to_process_articles_pass2.append(article)

    num_already_processed_pass2 = len(already_processed_pass2)
    num_to_process_pass2 = len(to_process_articles_pass2)

    print(f"Total articles loaded for Pass 2: {len(articles)}")
    print(f"Articles already processed for Pass 2: {num_already_processed_pass2}")
    print(f"Articles to be processed in Pass 2: {num_to_process_pass2}\n")
    logging.info(f"Total articles loaded for Pass 2: {len(articles)}")
    logging.info(f"Articles already processed for Pass 2: {num_already_processed_pass2}")
    logging.info(f"Articles to be processed for Pass 2: {num_to_process_pass2}\n")

    if num_to_process_pass2 == 0:
        print("No unprocessed articles found for Pass 2. Processing complete.")
        logging.info("No unprocessed articles found for Pass 2. Processing complete.")
    else:
        # Process each unprocessed article with a progress bar
        for idx, article in enumerate(tqdm(to_process_articles_pass2, desc="Pass 2: Assessing Sentiment and Fear")):
            # Determine the number of quotations
            quotation_keys = [key for key in article.keys() if key.startswith('quotation_')]
            num_quotations = len(quotation_keys)

            if num_quotations == 0:
                print(f"Article {idx + 1} has no quotations. Skipping...\n")
                logging.info(f"Article {idx + 1} has no quotations.")
                continue

            for i in range(1, num_quotations + 1):
                sentiment_field = f'sentiment_{i}'
                fear_field = f'fear_{i}'

                # Skip if already processed
                if sentiment_field in article and fear_field in article:
                    continue

                quotation = article.get(f'quotation_{i}', '').strip()
                if not quotation:
                    # If quotation is empty, set default values
                    article[sentiment_field] = 'neutral'
                    article[fear_field] = 'neutral'
                    logging.info(f"Article {idx + 1} - {sentiment_field}: neutral, {fear_field}: neutral (empty quotation)\n")
                    print(f"Article {idx + 1} - {sentiment_field}: neutral, {fear_field}: neutral (empty quotation)\n")
                    continue

                # Assess sentiment and fear
                assessment, input_tokens, output_tokens = assess_sentiment_fear(quotation)

                sentiment = assessment['sentiment']
                fear = assessment['fear']

                # Add assessments to the article
                article[sentiment_field] = sentiment
                article[fear_field] = fear

                # Log the results
                logging.info(f"Article {idx + 1} - {sentiment_field}: {sentiment}, {fear_field}: {fear}\n")
                print(f"Article {idx + 1} - {sentiment_field}: {sentiment}, {fear_field}: {fear}\n")
                print(f"Tokens Used for Assessment Article {idx + 1}-Quotation {i} - Input: {input_tokens}, Output: {output_tokens}\n")

    # Combine already processed and newly processed articles
    all_articles_pass2 = already_processed_pass2 + to_process_articles_pass2

    # Write the augmented data to the output JSON for Pass 2
    try:
        with open(OUTPUT_FILE_PASS2, 'w', encoding='utf-8') as outfile:
            json.dump(all_articles_pass2, outfile, ensure_ascii=False, indent=4)
        print(f"\nPass 2 complete. Output saved to '{OUTPUT_FILE_PASS2}'.")
        logging.info(f"Pass 2 complete. Output saved to '{OUTPUT_FILE_PASS2}'.")
    except Exception as e:
        print(f"Error writing to output file for Pass 2: {e}")
        logging.error(f"Error writing to output file for Pass 2: {e}")
        return

    # -------------------------- #
    #         Cost and Token Summary
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
