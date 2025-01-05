import json
import openai
import os
import sys
import uuid
import re
import logging
import builtins
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

# Override print so that everything printed to the terminal is also logged.
# (tqdm writes to stderr by default, so progress bars won't be logged.)
_original_print = builtins.print
def print(*args, **kwargs):
    """Print wrapper that logs everything going to stdout into processing.log."""
    if 'file' not in kwargs or kwargs['file'] is sys.stdout:
        text_to_log = " ".join(str(arg) for arg in args)
        logging.info(text_to_log)
    return _original_print(*args, **kwargs)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in .env or your environment.")

INPUT_FILE = 'distribution_set.jsonl'
BASE_OUTPUT_FILENAME = 'transmission_quantitative.jsonl'
DISCARDED_OUTPUT_FILENAME = 'transmission_quantitative_discarded.jsonl'

MODEL = 'gpt-4o'  # Example placeholder for a GPT-4-like model

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
    Returns 'distribution_subset_transmission_1.jsonl' if it doesn't exist, 
    else 'distribution_subset_transmission_2.jsonl', etc.
    If base_name itself is 'distribution_subset_transmission.jsonl', 
    we skip checking that exact file and look for suffixes.
    """
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]

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
    """Ensures the article has an 'id' field; if missing, create one."""
    if "id" not in article:
        article["id"] = default_prefix + str(uuid.uuid4())
    return article

def load_existing_output(filename):
    """Loads existing data from the given output file (JSONL)."""
    if not os.path.isfile(filename):
        return []

    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(record)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line in {filename}")
    return results

def write_jsonl(data_list, filename):
    """Utility to write a list of dicts (data_list) to filename as JSONL."""
    with open(filename, 'w', encoding='utf-8') as nf:
        for rec in data_list:
            nf.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ------------------------------ #
#  Second-Pass Verification      #
# ------------------------------ #

def verify_extracted_statements(article_id, fulltext, extracted_data):
    """
    1. Check each 'affirm_statement_n' or 'deny_statement_n' to see if it appears verbatim in 'fulltext'.
    2. If not found, remove it from 'extracted_data' and store it in a discard structure.
    3. Update 'affirm_count' or 'deny_count' accordingly.
    4. Return a dictionary with "discarded_statements" if any are removed.
    """
    discarded = []  # We'll store (key, statement) for each removed item

    # Affirm statements
    affirm_keys = sorted(k for k in extracted_data if k.startswith("affirm_statement_"))
    valid_affirms = []
    for key in affirm_keys:
        st = extracted_data[key]
        if st in fulltext:
            valid_affirms.append((key, st))
        else:
            # Mark for discard
            discarded.append((key, st))
    # Clear them from extracted_data
    for k in affirm_keys:
        del extracted_data[k]
    # Rebuild them with the correct numbering
    for i, (orig_key, st) in enumerate(valid_affirms, start=1):
        new_key = f"affirm_statement_{i}"
        extracted_data[new_key] = st
    extracted_data["affirm_count"] = len(valid_affirms)

    # Deny statements
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

    # Build a discard dictionary if we have any discards
    if discarded:
        discard_info = {
            "id": article_id,
            "fulltext": fulltext,
            "discarded_statements": [
                {
                    "key": k,         # e.g. "affirm_statement_2"
                    "statement": txt  # the verbatim text
                }
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
    Makes a single API call with NO built-in retries.
    """
    prompt = prompt_template.format(fulltext=fulltext)

    try:
        response = openai.ChatCompletion.create(
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
            max_tokens=16384,
        )
        print(response)

        reply = response['choices'][0]['message']['content'].strip()

        # Log the raw GPT response
        logging.info(f"Raw GPT response for article {article_id}:\n{reply}")

        # Remove code fences if present
        if reply.startswith("```") and reply.endswith("```"):
            reply = "\n".join(reply.split("\n")[1:-1]).strip()

        # Must be valid JSON
        if not (reply.startswith("{") and reply.endswith("}")):
            raise ValueError(f"Response is not a valid JSON object. Response was:\n{reply}")

        return json.loads(reply)

    except (OpenAIError, ValueError, json.JSONDecodeError) as e:
        logging.error("Detailed error during API call:", exc_info=True)
        print(f"\nERROR occurred during API call for Article ID {article_id}: {e}")
        return {"error": True}

def process_article_two_prompts(article_id, fulltext):
    """
    1. Runs TWO prompts (Affirm & Deny) on the same article text.
    2. Combines the results into one dict.
    3. Runs second-pass verification, returns:
       - final_data with verified statements
       - discard_info with any statements that didn't appear verbatim
    """
    if not fulltext.strip():
        return {"error": True}, None

    # Prompt #1 (Affirm)
    result1 = make_api_call(PROMPT_SPREAD_RISK_1, fulltext, article_id=article_id)

    # Prompt #2 (Deny)
    result2 = make_api_call(PROMPT_SPREAD_RISK_2, fulltext, article_id=article_id)

    if "error" in result1 or "error" in result2:
        return {"error": True}, None

    # Build final dict
    final_data = {}

    # Affirm
    if "affirm_count" in result1 and isinstance(result1["affirm_count"], int):
        final_data["affirm_count"] = result1["affirm_count"]
    else:
        final_data["affirm_count"] = 0
    for k, v in result1.items():
        if k.startswith("affirm_statement_"):
            final_data[k] = v

    # Deny
    if "deny_count" in result2 and isinstance(result2["deny_count"], int):
        final_data["deny_count"] = result2["deny_count"]
    else:
        final_data["deny_count"] = 0
    for k, v in result2.items():
        if k.startswith("deny_statement_"):
            final_data[k] = v

    # Second-pass verification
    discard_info = verify_extracted_statements(article_id, fulltext, final_data)

    return final_data, discard_info

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    1) Possibly create or continue in an existing output file.
    2) Load entire distribution_set.jsonl in order into 'all_articles', ensuring each has an ID.
    3) Immediately write all articles to the output file in input order (merging with existing data).
    4) Iterate over each article, run the two prompts, then do second-pass verification, update that article, 
       rewrite the entire file each time, and collect discards in a separate list.
    5) After completing all, write the discards to a separate JSONL file.
    """

    # --- Step 1: Check or create output file ---
    existing_output_data = []
    output_filename = BASE_OUTPUT_FILENAME

    if os.path.isfile(output_filename):
        # Load existing data
        existing_output_data = load_existing_output(output_filename)
        total_in_output = len(existing_output_data)
        print(f"Found existing {output_filename} with {total_in_output} records.")
        custom_msg = "Would you like to continue coding in this file? [yes/no] "
        print(custom_msg)
        choice = input().strip().lower()
        if choice.startswith('n'):
            custom_msg2 = "Create a new output file? [yes/no] "
            print(custom_msg2)
            choice2 = input().strip().lower()
            if choice2.startswith('y'):
                output_filename = get_next_output_file('distribution_subset_transmission')
                existing_output_data = []
                print(f"Using new output file: {output_filename}")
            else:
                sys.exit("Terminating script at user request.")
    else:
        print(f"No existing {output_filename}, starting fresh.")

    # --- Step 2: Load entire distribution set (in order) ---
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

    # Convert existing output into dict by ID for quick lookup
    existing_dict = {a["id"]: a for a in existing_output_data}

    # --- Step 3: Merge existing data with newly loaded articles ---
    final_list = []
    for article in all_articles:
        aid = article["id"]
        if aid in existing_dict:
            # Merge existing data onto the new article record
            merged_rec = {**article, **existing_dict[aid]}
            final_list.append(merged_rec)
        else:
            final_list.append(article)

    # Write all articles to the file right away
    write_jsonl(final_list, output_filename)
    print(f"Wrote {len(final_list)} articles (in order) to {output_filename}.")
    print("Now processing articles in real-time...")

    # Prepare a list to store discards
    discard_list = []

    # --- Step 4: Code each article with two prompts, rewriting file each time ---
    for i in tqdm(range(len(final_list)), desc="Coding Articles"):
        art = final_list[i]
        aid = art["id"]
        fulltext = art.get("fulltext", "")

        # Run the two-prompt extraction + verification
        result, discard_info = process_article_two_prompts(aid, fulltext)
        if "error" in result:
            # In case of error, do nothing more for this article
            continue

        # Clear any existing affirm/deny fields in case we re-run
        keys_to_clear = [
            k for k in art.keys()
            if k.startswith("affirm_statement_")
               or k.startswith("deny_statement_")
               or k in ["affirm_count", "deny_count"]
        ]
        for k in keys_to_clear:
            del art[k]

        # Merge new verified results
        for k, v in result.items():
            art[k] = v

        final_list[i] = art

        # If we have discard_info, add it to discard_list
        if discard_info is not None:
            discard_list.append(discard_info)

        # Rewrite entire file so it always remains up-to-date
        write_jsonl(final_list, output_filename)

    print(f"\nAll done! Final file: {output_filename}")

    # --- Step 5: Write the discards to a separate JSONL file ---
    if discard_list:
        print(f"Writing {len(discard_list)} discard records to {DISCARDED_OUTPUT_FILENAME}...")
        write_jsonl(discard_list, DISCARDED_OUTPUT_FILENAME)
    else:
        print("No discarded statements were found.")

    print("Done.")

if __name__ == "__main__":
    main()
