import json
import openai
import os
import sys
import logging
import builtins
import re
from dotenv import load_dotenv
from tqdm import tqdm
from openai.error import OpenAIError

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Override print so everything printed is also logged.
_original_print = builtins.print
def print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is sys.stdout:
        text_to_log = " ".join(str(arg) for arg in args)
        logging.info(text_to_log)
    return _original_print(*args, **kwargs)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in .env or your environment.")

INPUT_FILE = 'distribution_set.jsonl'
OUTPUT_FILE = 'distribution_set_final.jsonl'
MODEL = 'gpt-4o'  # Using gpt-4o per your requirement

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
#       OpenAI API Helper        #
# ------------------------------ #

def call_clean_api(text_block):
    """
    Calls the OpenAI API with the cleaning instructions and returns the cleaned text.
    Returns None if there's an error or if the model doesn't return valid JSON.
    """
    prompt = CLEAN_PROMPT_TEMPLATE.format(text_block=text_block)

    try:
        response = openai.ChatCompletion.create(
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
            max_tokens=16384  # Set to 16384, as requested
        )
        print(response)

        reply = response['choices'][0]['message']['content'].strip()
        logging.info(f"Raw GPT response:\n{reply}")

        # ------------------------------
        # Remove any leading/trailing code fences if present
        # e.g. ```json ... ```
        # This helps pass the valid JSON check below.
        # ------------------------------
        # A simple approach using a regex to strip triple backticks from start/end
        reply = re.sub(r"^```(?:json)?", "", reply)
        reply = re.sub(r"```$", "", reply)
        reply = reply.strip()

        # Must be valid JSON
        if not (reply.startswith("{") and reply.endswith("}")):
            raise ValueError(f"Response is not a valid JSON object. Got:\n{reply}")

        cleaned = json.loads(reply)
        if "corrected_text" not in cleaned:
            raise ValueError("No 'corrected_text' field returned in JSON.")
        return cleaned["corrected_text"]

    except (OpenAIError, ValueError, json.JSONDecodeError) as e:
        logging.error("Error during text cleaning API call:", exc_info=True)
        print(f"ERROR occurred during text cleaning API call: {e}")
        return None

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
    1) Possibly continue from an existing distribution_set_final.jsonl 
       if user chooses not to overwrite.
    2) Load the set of already-processed IDs from distribution_set_final.jsonl.
    3) For each record in distribution_set.jsonl, 
       if its ID is not in the processed set:
         - Attempt to clean the 'fulltext' field with the LLM.
         - If error, set 'error'='yes'.
         - Mark 'processed'='yes'.
         - Write the updated record to distribution_set_final.jsonl immediately.
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
        cleaned_text = call_clean_api(original_text)

        if cleaned_text is None:
            # If error or invalid response, keep the original text
            # and add "error": "yes"
            entry["error"] = "yes"
            cleaned_text = original_text
        else:
            entry["error"] = "no"

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
