import json
import openai
import os
import sys
import uuid
import re
import logging
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

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in .env or your environment.")

INPUT_FILE = 'distribution_set.jsonl'
BASE_OUTPUT_FILENAME = 'distribution_subset_transmission.jsonl'

MODEL = 'gpt-4o'  # Example model reference

# ------------------------------ #
#  Two PROMPTS (Affirm / Deny)   #
# ------------------------------ #
# Use double curly braces in JSON snippet to avoid .format() conflicts.

PROMPT_SPREAD_RISK_1 = """
You are an analytical assistant. Given the article text below, determine if it suggests, implies, or explicitly states that bird flu has potential to:
1. Cause a public health crisis,
2. Evolve become more transmissible or infectious to humans,
3. Evolve to become more deadly to humans,
4. Cause a pandemic,
5. Or start spreading human to human.

If any of these five criteria are met, then "spread_risk" = "yes".
Otherwise, "spread_risk" = "no".

Additionally, for the "feedback" field, provide TWO things:
1) Which criterion or criteria above that the article fulfills, if any;
2) All exact quotations from the article text (verbatim) that led to your determination.

Respond ONLY with a JSON object in the following form:

{{
  "spread_risk": "yes" or "no",
  "feedback": "Which criterion/criteria are fulfilled (if any) + verbatim quotations from the text"
}}

Article Text:
{fulltext}
"""

PROMPT_SPREAD_RISK_2 = """
You are an analytical assistant. Given the article text below, determine if it explicitly downplays, minimizes, or otherwise suggests that the risk of bird flu is low of:
1. Causing a public health crisis,
2. Evolving become more transmissible or infectious to humans,
3. Evolving to become more deadly to humans,
4. Causing a pandemic,
5. Or starting to spread human to human.

If any of these five criteria are met, then "spread_risk" = "yes".
Otherwise, "spread_risk" = "no".

Additionally, for the "feedback" field, provide TWO things:
1) Which criterion or criteria above that the article fulfills, if any;
2) All exact quotations from the article text (verbatim) that led to your determination.

Respond ONLY with a JSON object in the following form:

{{
  "spread_risk": "yes" or "no",
  "feedback": "Which criterion/criteria are fulfilled (if any) + verbatim quotations from the text"
}}

Article Text:
{fulltext}
"""

# ------------------------------ #
#    (Commented) Versioning     #
# ------------------------------ #
"""
def find_latest_output_file(base_prefix):
    ...
def get_next_output_file(base_prefix):
    ...
def compare_spread_fields(old_file, new_file):
    ...
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

def is_coded(article):
    """An article is 'coded' if spread_risk is 'yes' or 'no'."""
    return article.get("spread_risk") in ["yes", "no"]

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
#   Chunking & Classification    #
# ------------------------------ #

def chunk_text(text, chunk_size=3000):
    """
    Splits 'text' into chunks of up to 'chunk_size' characters each.
    Returns a list of chunks.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks


def make_api_call_once(prompt_template, fulltext, article_id=None):
    """
    Makes ONE attempt to call the API for classification.
    Raises ValueError if the JSON is incomplete or invalid.
    """
    prompt = prompt_template.format(fulltext=fulltext)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an analytical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096,
    )
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


def make_api_call(prompt_template, fulltext, article_id=None):
    """
    Wraps make_api_call_once in a short retry loop (up to 3 attempts).
    If we detect partial JSON, we append a note to the fulltext:
      "Your previous response was not valid JSON. Please provide valid JSON."
    """
    fix_note = ""
    attempts = 3

    for attempt in range(attempts):
        try:
            combined_text = fulltext + fix_note
            return make_api_call_once(prompt_template, combined_text, article_id=article_id)
        except (OpenAIError, ValueError, json.JSONDecodeError) as e:
            logging.error("Detailed error during API call:", exc_info=True)
            print(f"\nERROR occurred for Article ID {article_id} (Attempt {attempt+1}): {e}")
            
            if attempt < attempts - 1:
                # Nudge the model on subsequent attempts
                fix_note = "\n\nYour previous response was not valid JSON. Please provide valid JSON."
                continue
            else:
                # Final attempt failed
                choice = input("Do you want to continue? [yes/no] ").strip().lower()
                if choice.startswith('n'):
                    sys.exit("Terminating script at user request due to repeated API error.")
                # Return an 'error' coded item
                return {
                    "spread_risk": "error",
                    "feedback": "API call failed or invalid response after 3 attempts."
                }


def process_article_chunk(chunk_text_str, article_id, prompt_1, prompt_2):
    """
    Runs the two prompts on a single chunk of text.
    Returns {"spread_risk": "yes"/"no"/"error", "feedback": "..."} for this chunk.
    """
    # Call #1
    result1 = make_api_call(prompt_1, chunk_text_str, article_id=article_id)
    spread1 = result1.get("spread_risk", "error")
    fb1 = result1.get("feedback", "")

    # Call #2
    result2 = make_api_call(prompt_2, chunk_text_str, article_id=article_id)
    spread2 = result2.get("spread_risk", "error")
    fb2 = result2.get("feedback", "")

    if spread1 == "yes" or spread2 == "yes":
        final_spread = "yes"
    else:
        # If both are 'no' or any 'error'
        if spread1 == "error" or spread2 == "error":
            final_spread = "error"
        else:
            final_spread = "no"

    final_feedback = f"[CHUNK FEEDBACK]\nAFFIRMS RISK: {fb1}\nDENIES RISK: {fb2}"
    return {
        "spread_risk": final_spread,
        "feedback": final_feedback
    }


def process_article_two_prompts_in_chunks(fulltext, article_id=None):
    """
    Chunk the fulltext if it's too large. For each chunk:
      - get 'spread_risk' + 'feedback'
    Then combine across all chunks:
      - If any chunk is "yes", overall is "yes"
      - Concatenate feedback from all chunks
    """
    chunks = chunk_text(fulltext, chunk_size=3000)

    overall_spread = "no"
    all_feedbacks = []

    for ctext in chunks:
        chunk_result = process_article_chunk(
            ctext,
            article_id,
            PROMPT_SPREAD_RISK_1,
            PROMPT_SPREAD_RISK_2
        )
        if chunk_result["spread_risk"] == "yes":
            overall_spread = "yes"
        elif chunk_result["spread_risk"] == "error" and overall_spread != "yes":
            overall_spread = "error"

        all_feedbacks.append(chunk_result["feedback"])

    combined_feedback = "\n\n".join(all_feedbacks)
    return {
        "spread_risk": overall_spread,
        "feedback": combined_feedback
    }

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    1) Possibly create or continue in an existing output file. 
       - If existing is fully coded, prompt to create new or terminate.
       - If existing is partially coded, prompt to continue or create new or terminate.
    2) Load entire distribution_set.jsonl in order into 'all_articles', ensuring each has an ID.
    3) Immediately write all articles to the output file in input order. 
       - For articles that are already coded in the existing file, fill them in. 
       - For uncoded ones, set "spread_risk":"unknown","spread_feedback":"unknown".
    4) For each uncoded article, chunk the text if needed, run two prompts per chunk,
       unify the results, update that article in memory, then rewrite the entire file each time.
    """

    # --- Step 1: Check or create output file ---
    existing_output_data = []
    output_filename = BASE_OUTPUT_FILENAME

    if os.path.isfile(output_filename):
        # Load existing data
        existing_output_data = load_existing_output(output_filename)
        coded_count = sum(is_coded(a) for a in existing_output_data)
        total_in_output = len(existing_output_data)
        print(f"Found existing {output_filename} with {total_in_output} records, "
              f"{coded_count} of which are fully coded.")

        if total_in_output > 0 and coded_count == total_in_output:
            # All entries coded
            choice = input(
                "All records in this file are coded. Create a new output file? [yes/no] "
            ).strip().lower()
            if choice.startswith('y'):
                output_filename = get_next_output_file('distribution_subset_transmission')
                existing_output_data = []  # new file, start fresh
                print(f"Using new output file: {output_filename}")
            else:
                sys.exit("Terminating script at user request. (All are coded)")
        else:
            # Not all coded
            choice = input(
                "Not all records are coded in this file. Continue coding here? [yes/no] "
            ).strip().lower()
            if choice.startswith('n'):
                # Ask if we want to create a new file or terminate
                choice2 = input("Create a new output file? [yes/no] ").strip().lower()
                if choice2.startswith('y'):
                    output_filename = get_next_output_file('distribution_subset_transmission')
                    existing_output_data = []  # brand new file
                    print(f"Using new output file: {output_filename}")
                else:
                    sys.exit("Terminating script at user request. (Not all coded)")
    else:
        # No existing file, so we start fresh
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

    # --- Step 3: Immediately create a 'final_list' in input order ---
    final_list = []
    for article in all_articles:
        aid = article["id"]
        if aid in existing_dict:
            final_list.append(existing_dict[aid])
        else:
            article["spread_risk"] = article.get("spread_risk", "unknown")
            article["spread_feedback"] = article.get("spread_feedback", "unknown")
            final_list.append(article)

    # Write all articles to the new (or existing) file right away
    write_jsonl(final_list, output_filename)
    print(f"Wrote {len(final_list)} articles (in order) to {output_filename}.")
    print("Now coding uncoded articles in real-time...")

    # --- Step 4: Code each uncoded article, rewriting file each time ---
    uncoded_indices = [i for i, art in enumerate(final_list) if not is_coded(art)]
    for i in tqdm(uncoded_indices, desc="Coding Articles"):
        art = final_list[i]
        aid = art["id"]
        fulltext = art.get("fulltext", "")

        # If still not coded, do chunk-based classification
        if not is_coded(art):
            result = process_article_two_prompts_in_chunks(fulltext, article_id=aid)
            art["spread_risk"] = result["spread_risk"]
            art["spread_feedback"] = result["feedback"]
            final_list[i] = art

            # Rewrite entire file so it always remains up-to-date
            write_jsonl(final_list, output_filename)

    print(f"\nAll done! Final file: {output_filename}")

if __name__ == "__main__":
    main()
