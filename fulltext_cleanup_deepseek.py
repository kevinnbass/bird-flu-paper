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

# Updated input/output
INPUT_FILE = 'distribution_set_final.jsonl'
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
        # ensure_ascii=False to keep Unicode chars intact
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ------------------------------ #
#       DeepSeek API Helper      #
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
            max_tokens=8192
        )

        # Debug: Log the raw API response
        reply = response.choices[0].message.content.strip()
        logging.info(f"Raw API response for article {article_id}:\n{reply}")

        # Remove any code fences if present
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
#            Main App            #
# ------------------------------ #

def main():
    """
    Reads distribution_set_final.jsonl line by line. 
    For each record:
      - If "error":"yes", call the LLM to fix 'fulltext'. 
        * If success, set "error":"no".
        * If an error occurs, keep "error":"yes".
      - If "error" is not "yes", copy the record as is.
    Writes all records (in the same order) to distribution_set_final_deepseek.jsonl.
    """
    # Start fresh (overwrite) the output file if it exists
    if os.path.isfile(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Read the input file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        return

    print(f"Loaded {len(lines)} lines from {INPUT_FILE}. Beginning cleaning process...")

    processed_count = 0
    total_count = 0

    for line in tqdm(lines, desc="Processing"):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping invalid JSON line in input file.")
            continue

        total_count += 1

        # Only process if error == "yes"
        if entry.get("error") == "yes":
            _id = entry.get("id", "unknown_id")
            original_text = entry.get("fulltext", "")

            api_response = make_api_call(original_text, article_id=_id)

            if api_response.get("error"):
                # If the API call failed, we keep error="yes"
                # Keep the text as-is
                pass
            else:
                # If successful, set "error" to "no" and replace fulltext
                entry["error"] = "no"
                cleaned_text = api_response.get("corrected_text", original_text)
                entry["fulltext"] = cleaned_text

            processed_count += 1

        # Write the (possibly updated) record to output
        append_jsonl(entry, OUTPUT_FILE)

    print(f"\nDone. Processed {processed_count} record(s) that had error='yes'.")
    print(f"Total records in input: {total_count}")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
