import json
import os
import sys
import uuid
import re
import logging
import builtins
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI  # Updated import for DeepSeek API

# Use a generic exception for DeepSeek errors
DeepSeekError = Exception

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

logging.basicConfig(
    filename='processing_deepseek.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Override print to log output
_original_print = builtins.print
def print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is sys.stdout:
        text_to_log = " ".join(str(arg) for arg in args)
        logging.info(text_to_log)
    return _original_print(*args, **kwargs)

# Load environment variables
load_dotenv()

# Initialize OpenAI client for DeepSeek
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# Check if the API key is set
if not os.getenv('DEEPSEEK_API_KEY'):
    raise ValueError("DeepSeek API key not found. Please set it in .env or your environment.")

INPUT_FILE = 'distribution_set.jsonl'
OUTPUT_FILE = 'distribution_set_final_deepseek.jsonl'
MODEL = 'deepseek-chat'  # Updated to use the "deepseek-chat" model

# ------------------------------ #
#         Prompt Template        #
# ------------------------------ #

CLEAN_PROMPT_TEMPLATE = """
You are a text cleaning assistant. 
Please fix the following text by:
1) Replacing all � with the appropriate punctuation marks (apostrophes, quotes, etc.). 
2) Replacing all \" with the correct punctuation or symbol. 
3) Removing or replacing any <a></a> tags. 
4) Do not introduce new paragraphs or line breaks; keep everything in a single line.
5) Return only JSON with one key: "corrected_text". 
   Make sure not to add any extra fields or commentary.

Text to clean:
{text_block}
"""

# ------------------------------ #
#      Helper: JSONL Writer      #
# ------------------------------ #

def append_jsonl(record, filename):
    """Appends a single JSON record to filename as JSONL (one record per line)."""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ------------------------------ #
#       DeepSeek API Helper       #
# ------------------------------ #

def make_api_call(text_block, article_id=None):
    """
    Makes an API call to the DeepSeek model using the provided prompt template and text block.
    Returns the parsed JSON response or an error dictionary.
    """
    prompt = CLEAN_PROMPT_TEMPLATE.format(text_block=text_block)
    try:
        # Debug: Log the prompt being sent to the API
        logging.info(f"Sending API request for article {article_id} with prompt:\n{prompt}")

        # Make the API call
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a text cleaning assistant. You will strictly follow the user's instructions "
                        "on how to fix the text and return only valid JSON with the corrected text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,  # Keep it deterministic
            max_tokens=8192  # Set to 16384, as requested
        )

        # Debug: Log the raw API response
        reply = response.choices[0].message.content.strip()
        logging.info(f"Raw API response for article {article_id}:\n{reply}")

        # Remove code fences if present
        if reply.startswith("```") and reply.endswith("```"):
            reply = "\n".join(reply.split("\n")[1:-1]).strip()

        # Ensure the response is valid JSON
        if not (reply.startswith("{") and reply.endswith("}")):
            raise ValueError(f"Response is not a valid JSON object. Response was:\n{reply}")

        return json.loads(reply)

    except DeepSeekError as dse:
        logging.error(f"DeepSeek API error for article {article_id}: {dse}", exc_info=True)
        return {"error": True}
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decode error for article {article_id}: {jde}", exc_info=True)
        return {"error": True}
    except Exception as e:
        logging.error(f"Unexpected error for article {article_id}: {e}", exc_info=True)
        return {"error": True}

# ------------------------------ #
#  Read IDs Already Processed    #
# ------------------------------ #

def load_processed_ids(filename):
    """
    Reads the output file (if it exists) and collects a set of IDs 
    that have 'processed' = 'yes'. This allows us to skip them 
    if we want to continue after interruption.
    """
    processed_set = set()
    if not os.path.isfile(filename):
        return processed_set  # Empty set if file doesn't exist

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # If the record has processed="yes", store the ID
                if rec.get("processed") == "yes":
                    # We assume each record has a unique 'id'
                    _id = rec.get("id")
                    if _id:
                        processed_set.add(_id)
            except json.JSONDecodeError:
                continue
    return processed_set

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    1) Possibly continue from an existing distribution_set_final_deepseek.jsonl 
       if user chooses not to overwrite.
    2) Load the set of already-processed IDs from distribution_set_final_deepseek.jsonl.
    3) For each record in distribution_set.jsonl, 
       if its ID is not in the processed set:
         - Attempt to clean the 'fulltext' field with the LLM.
         - If error, set 'error'='yes'.
         - Mark 'processed'='yes'.
         - Write the updated record to distribution_set_final_deepseek.jsonl immediately.
    """

    # Check if the output file already exists
    processed_ids = set()
    if os.path.isfile(OUTPUT_FILE):
        print(f"Output file '{OUTPUT_FILE}' already exists.")
        print("Do you want to overwrite it and start fresh, or continue? [overwrite/continue]")
        choice = input().strip().lower()
        if choice.startswith('o'):
            os.remove(OUTPUT_FILE)
            print(f"Deleted existing {OUTPUT_FILE}, will create fresh.")
        else:
            print("Continuing. Loading already processed IDs...")
            processed_ids = load_processed_ids(OUTPUT_FILE)
            print(f"Found {len(processed_ids)} records already processed.")
    else:
        print("No existing output file, starting fresh.")

    # Read the input file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        return

    print(f"Loaded {len(lines)} lines from {INPUT_FILE}. Beginning cleaning process...")

    processed_count = 0
    to_process_count = 0

    # Loop over each line and process if not already processed
    for line in tqdm(lines, desc="Cleaning Text"):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping invalid JSON line in input file.")
            continue

        # If there's no ID, skip or generate one, but best is to skip or assign
        _id = entry.get("id")
        if not _id:
            logging.warning("Skipping record with no 'id' field.")
            continue

        # Check if already processed
        if _id in processed_ids:
            # Already done, skip
            continue

        to_process_count += 1

        # Extract the 'fulltext' field
        original_text = entry.get("fulltext", "")

        # Call the API to clean it
        api_response = make_api_call(original_text, article_id=_id)

        if api_response.get("error"):
            # If error or invalid response, keep the original text
            # and add "error": "yes"
            entry["error"] = "yes"
            cleaned_text = original_text
        else:
            entry["error"] = "no"
            cleaned_text = api_response.get("corrected_text", original_text)

        # Mark this entry as processed so we don't re-run it in a subsequent pass
        entry["processed"] = "yes"
        # Replace the old fulltext with the cleaned version
        entry["fulltext"] = cleaned_text

        # Write to output JSONL in real-time
        append_jsonl(entry, OUTPUT_FILE)
        processed_count += 1
        processed_ids.add(_id)  # So we don't do it again if script restarts

    print(f"\nCompleted cleaning process.")
    print(f"Processed {processed_count} new record(s) (skipped {len(processed_ids) - processed_count} already processed).")
    print(f"Of those, {to_process_count} were unprocessed in the input file, and {processed_count} got done this run.")
    print(f"Final output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()