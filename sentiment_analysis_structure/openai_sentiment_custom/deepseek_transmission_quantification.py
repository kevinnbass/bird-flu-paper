import json
import os
import sys
import uuid
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

INPUT_FILE = 'distribution_set_final.jsonl'
BASE_OUTPUT_FILENAME = 'transmission_quantitative_deepseek.jsonl'
DISCARDED_OUTPUT_FILENAME = 'transmission_quantitative_discarded_deepseek.jsonl'

MODEL = 'deepseek-chat'  # Updated to use the "deepseek-chat" model

# ------------------------------ #
#  Two PROMPTS (Affirm / Deny)   #
# ------------------------------ #

PROMPT_SPREAD_RISK_1 = """
You are an analytical assistant. 
Step 1: Systematically check the article text below for any statements (or groups of statements) that individually or together suggest, imply, or explicitly state that bird flu has potential to:

  (a) Cause a public health crisis,
  (b) Evolve to become more transmissible or infectious to humans,
  (c) Evolve to become more deadly to humans,
  (d) Cause a pandemic,
  (e) Start to spread human to human.

  (f) Also, check for any statement using the word "pandemic", "coronavirus", "Covid", or "COVID-19" that ***DOES NOT DENY*** that bird flu might be similar.

Step 2: For EACH such statement you find, return it in a unique "affirm_statement_n" field (verbatim text, no summarization).

Step 3: Count how many total statements you found. Put that number in "affirm_count".

Step 4: If no statements are found, include "affirm_count": 0 and no "affirm_statement_n" fields.

Respond ONLY with a JSON object in the format:

{{
  "affirm_count": <integer>,
  "affirm_statement_1": "...",
  "affirm_statement_2": "...",
  ...
}}

Article Text:
{fulltext}
"""

PROMPT_SPREAD_RISK_2 = """
You are an analytical assistant. 
Step 1: Systematically check the article text below for any statements that explicitly downplay, minimize, or otherwise suggest that the risk of bird flu is low in terms of:
  (a) Causing a public health crisis,
  (b) Evolving to become more transmissible or infectious to humans,
  (c) Evolving to become more deadly to humans,
  (d) Causing a pandemic,
  (e) Starting to spread human to human.

  (f) Also, check for any statement using the word "pandemic", "coronavirus", "Covid", or "COVID-19" that ***DENIES*** that bird flu might be similar.

Step 2: For EACH such statement you find, return it in a unique "deny_statement_n" field (verbatim text, no summarization).

Step 3: Count how many total statements you found. Put that number in "deny_count".

Step 4: If no statements are found, include "deny_count": 0 and no "deny_statement_n" fields.

Respond ONLY with a JSON object in the format:

{{
  "deny_count": <integer>,
  "deny_statement_1": "...",
  "deny_statement_2": "...",
  ...
}}

Article Text:
{fulltext}
"""

# ------------------------------ #
#    Helper: Next File Number    #
# ------------------------------ #

def get_next_output_file(base_name='distribution_subset_transmission'):
    """
    Returns the next available filename in the sequence (e.g., base_name_1.jsonl, base_name_2.jsonl).
    """
    n = 1
    while True:
        candidate = f"{base_name}_{n}.jsonl"
        if not os.path.exists(candidate):
            return candidate
        n += 1

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def assign_id_if_missing(article, default_prefix="article_"):
    """
    Ensures the article has an 'id' field. If missing, generates a unique ID.
    """
    if "id" not in article:
        article["id"] = default_prefix + str(uuid.uuid4())
    return article

def load_existing_output(filename):
    """
    Loads existing data from a JSONL file. Returns an empty list if the file doesn't exist or is empty.
    """
    if not os.path.isfile(filename):
        return []
    if os.path.getsize(filename) == 0:
        logging.warning(f"Input file {filename} is empty.")
        return []
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(record)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line {line_num} in {filename}")
    return results

def write_jsonl(data_list, filename):
    """
    Writes a list of dictionaries to a JSONL file.
    """
    with open(filename, 'w', encoding='utf-8') as nf:
        for rec in data_list:
            nf.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------------------ #
#  Second-Pass Verification      #
# ------------------------------ #

def verify_extracted_statements(article_id, fulltext, extracted_data):
    """
    Verifies that extracted statements appear verbatim in the fulltext.
    Removes invalid statements and updates counts.
    """
    discarded = []
    affirm_keys = sorted(k for k in extracted_data if k.startswith("affirm_statement_"))
    valid_affirms = []
    for key in affirm_keys:
        st = extracted_data[key]
        if st in fulltext:
            valid_affirms.append((key, st))
        else:
            discarded.append((key, st))
    for k in affirm_keys:
        del extracted_data[k]
    for i, (orig_key, st) in enumerate(valid_affirms, start=1):
        new_key = f"affirm_statement_{i}"
        extracted_data[new_key] = st
    extracted_data["affirm_count"] = len(valid_affirms)

    deny_keys = sorted(k for k in extracted_data if k.startswith("deny_statement_"))
    valid_denies = []
    for key in deny_keys:
        st = extracted_data[key]
        if st in fulltext:
            valid_denies.append((key, st))
        else:
            discarded.append((key, st))
    for k in deny_keys:
        del extracted_data[k]
    for i, (orig_key, st) in enumerate(valid_denies, start=1):
        new_key = f"deny_statement_{i}"
        extracted_data[new_key] = st
    extracted_data["deny_count"] = len(valid_denies)

    if discarded:
        discard_info = {
            "id": article_id,
            "fulltext": fulltext,
            "discarded_statements": [
                {"key": k, "statement": txt}
                for (k, txt) in discarded
            ]
        }
        return discard_info
    else:
        return None

# ------------------------------ #
#       Make API Call            #
# ------------------------------ #

def make_api_call(prompt_template, fulltext, article_id=None):
    """
    Makes an API call to the DeepSeek model using the provided prompt template and fulltext.
    Returns the parsed JSON response or an error dictionary.
    """
    prompt = prompt_template.format(fulltext=fulltext)
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
                        "You are an analytical assistant extracting verbatim statements. "
                        "Follow the user's instructions carefully. Output valid JSON. "
                        "Do not add extra commentary or fields."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=8192,
            stream=False
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
#  Process Article with Two Prompts #
# ------------------------------ #

def process_article_two_prompts(article_id, fulltext):
    """
    Processes an article using both prompts (Affirm and Deny).
    Returns the combined results and any discarded statements.
    """
    if not fulltext.strip():
        logging.warning(f"Article {article_id} has empty fulltext. Skipping.")
        return {"error": True}, None

    # Call the first prompt (Affirm)
    result1 = make_api_call(PROMPT_SPREAD_RISK_1, fulltext, article_id=article_id)
    if "error" in result1:
        logging.error(f"Error processing affirm prompt for article {article_id}.")
        return {"error": True}, None

    # Call the second prompt (Deny)
    result2 = make_api_call(PROMPT_SPREAD_RISK_2, fulltext, article_id=article_id)
    if "error" in result2:
        logging.error(f"Error processing deny prompt for article {article_id}.")
        return {"error": True}, None

    # Combine results
    final_data = {}

    # Affirm results
    if "affirm_count" in result1 and isinstance(result1["affirm_count"], int):
        final_data["affirm_count"] = result1["affirm_count"]
    else:
        final_data["affirm_count"] = 0
    for k, v in result1.items():
        if k.startswith("affirm_statement_"):
            final_data[k] = v

    # Deny results
    if "deny_count" in result2 and isinstance(result2["deny_count"], int):
        final_data["deny_count"] = result2["deny_count"]
    else:
        final_data["deny_count"] = 0
    for k, v in result2.items():
        if k.startswith("deny_statement_"):
            final_data[k] = v

    # Verify extracted statements
    discard_info = verify_extracted_statements(article_id, fulltext, final_data)

    return final_data, discard_info

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    Main function to process articles, handle user input, and manage output files.
    """
    existing_output_data = []
    output_filename = BASE_OUTPUT_FILENAME
    if os.path.isfile(output_filename):
        existing_output_data = load_existing_output(output_filename)
        total_in_output = len(existing_output_data)
        print(f"Found existing {output_filename} with {total_in_output} records.")
        choice = input("Would you like to continue coding in this file? [yes/no] ").strip().lower()
        if choice.startswith('n'):
            choice2 = input("Create a new output file? [yes/no] ").strip().lower()
            if choice2.startswith('y'):
                output_filename = get_next_output_file('distribution_subset_transmission')
                existing_output_data = []
                print(f"Using new output file: {output_filename}")
            else:
                sys.exit("Terminating script at user request.")
    else:
        print(f"No existing {output_filename}, starting fresh.")
    all_articles = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                    article = assign_id_if_missing(article)
                    all_articles.append(article)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line in input file.")
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        return
    print(f"Loaded {len(all_articles)} articles from {INPUT_FILE}.")
    existing_dict = {a["id"]: a for a in existing_output_data}
    final_list = []
    for article in all_articles:
        aid = article["id"]
        if aid in existing_dict:
            merged_rec = {**article, **existing_dict[aid]}
            final_list.append(merged_rec)
        else:
            final_list.append(article)
    write_jsonl(final_list, output_filename)
    print(f"Wrote {len(final_list)} articles (in order) to {output_filename}.")
    print("Now processing articles in real-time...")
    discard_list = []
    try:
        for i in tqdm(range(len(final_list)), desc="Coding Articles"):
            art = final_list[i]
            aid = art["id"]
            # Check for "fulltext" key
            if "fulltext" not in art:
                art["processing_error"] = "Missing fulltext"
                logging.warning(f"Article {aid} is missing 'fulltext' key. Adding processing error.")
                continue
            fulltext = art["fulltext"]
            result, discard_info = process_article_two_prompts(aid, fulltext)
            if "error" in result:
                art["api_error"] = True
                continue
            keys_to_clear = [k for k in art.keys() if k.startswith("affirm_statement_") or k.startswith("deny_statement_") or k in ["affirm_count", "deny_count"]]
            for k in keys_to_clear:
                del art[k]
            for k, v in result.items():
                art[k] = v
            final_list[i] = art
            if discard_info is not None:
                discard_list.append(discard_info)
            write_jsonl(final_list, output_filename)
        print(f"\nAll done! Final file: {output_filename}")
        if discard_list:
            print(f"Writing {len(discard_list)} discard records to {DISCARDED_OUTPUT_FILENAME}...")
            write_jsonl(discard_list, DISCARDED_OUTPUT_FILENAME)
        else:
            print("No discarded statements were found.")
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user. Saving current state.")
        write_jsonl(final_list, output_filename)
        print("Current state saved. Exiting.")
    print("Done.")

if __name__ == "__main__":
    main()
