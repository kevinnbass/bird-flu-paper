import json
import openai
import os
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

INPUT_FILE = 'distribution_mortality_subset.jsonl'
MODEL = 'gpt-4o'  # As requested, no retries, using 'gpt-4o' only.

# ------------------------------ #
#  Two PROMPTS (Affirm / Deny)   #
# ------------------------------ #

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
#      Versioning Functions      #
# ------------------------------ #

def find_latest_output_file(base_prefix="spread_subset_test"):
    """
    Find all files named 'spread_subset_test_*.jsonl' in the current directory.
    Return the one with the highest integer suffix, or None if none found.
    """
    files = [f for f in os.listdir('.') if f.startswith(base_prefix + '_') and f.endswith('.jsonl')]
    max_num = 0
    latest_file = None
    pattern = re.compile(rf'^{base_prefix}_(\d+)\.jsonl$')
    for fname in files:
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_file = fname
    return latest_file

def get_next_output_file(base_prefix="spread_subset_test"):
    """
    Starting from suffix 2, find the first 'spread_subset_test_{n}.jsonl' that doesn't exist.
    If 2 exists, try 3, etc.
    """
    n = 2
    while True:
        candidate = f"{base_prefix}_{n}.jsonl"
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

def make_api_call(prompt_template, fulltext):
    """
    Makes a single API call with NO retries.
    Logs the full exception details if something goes wrong.
    Returns a dict with "spread_risk" and "feedback".
    """
    prompt = prompt_template.format(fulltext=fulltext)
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an analytical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
        )
        reply = response['choices'][0]['message']['content'].strip()

        # Remove code fences if present
        if reply.startswith("```") and reply.endswith("```"):
            reply = "\n".join(reply.split("\n")[1:-1]).strip()

        # Must be valid JSON
        if not (reply.startswith("{") and reply.endswith("}")):
            raise ValueError(f"Response is not a valid JSON object. Response was:\n{reply}")

        result = json.loads(reply)
        return result

    except (OpenAIError, ValueError, json.JSONDecodeError) as e:
        # Expanded logging for better debugging
        logging.error("Detailed error during API call:", exc_info=True)
        return {
            "spread_risk": "error",
            "feedback": "API call failed or invalid response."
        }

def process_article_two_prompts(fulltext):
    """
    Runs TWO prompts on the same article text.

    - Call #1: PROMPT_SPREAD_RISK_1 (the 'affirm' type).
    - Call #2: PROMPT_SPREAD_RISK_2 (the 'deny' type).

    Then:
      - If either call returns "yes", the final 'spread_risk' is "yes".
      - Otherwise, "no".

    The final 'feedback' is a combination:
      "AFFIRMS RISK: {call1_feedback} DENIES RISK: {call2_feedback}"
    """
    if not fulltext.strip():
        return {
            "spread_risk": "error",
            "feedback": "No text provided."
        }

    # Call #1
    result1 = make_api_call(PROMPT_SPREAD_RISK_1, fulltext)
    spread1 = result1.get("spread_risk", "error")
    fb1 = result1.get("feedback", "")

    # Call #2
    result2 = make_api_call(PROMPT_SPREAD_RISK_2, fulltext)
    spread2 = result2.get("spread_risk", "error")
    fb2 = result2.get("feedback", "")

    # Determine final spread_risk
    # If either is "yes", overall is "yes"
    if spread1 == "yes" or spread2 == "yes":
        final_spread = "yes"
    else:
        # If neither is "yes", or both are "error"/"no", final is "no"
        final_spread = "no"

    # Combine feedback
    final_feedback = f"AFFIRMS RISK: {fb1} DENIES RISK: {fb2}"

    return {
        "spread_risk": final_spread,
        "feedback": final_feedback
    }

def compare_spread_fields(old_file, new_file):
    """
    Compares the 'spread_risk' field in old_file vs new_file by matching 'id'.
    Returns two lists:
      diffs:    list of IDs whose 'spread_risk' differ
      same_ids: list of IDs whose 'spread_risk' is the same
    Also returns sets of all old IDs and all new IDs for reference.
    """
    old_data = {}
    new_data = {}

    old_ids = set()
    new_ids = set()

    # Read old
    if os.path.isfile(old_file):
        with open(old_file, 'r', encoding='utf-8') as f_old:
            for line in f_old:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    aid = record.get("id")
                    if aid:
                        old_data[aid] = record
                        old_ids.add(aid)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in old file.")

    # Read new
    if os.path.isfile(new_file):
        with open(new_file, 'r', encoding='utf-8') as f_new:
            for line in f_new:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    aid = record.get("id")
                    if aid:
                        new_data[aid] = record
                        new_ids.add(aid)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in new file.")

    diffs = []
    same_ids_list = []
    # Compare only IDs present in BOTH
    for aid in new_data:
        if aid in old_data:
            old_spread = old_data[aid].get("spread_risk")
            new_spread = new_data[aid].get("spread_risk")
            if old_spread != new_spread:
                diffs.append(aid)
            else:
                same_ids_list.append(aid)

    return diffs, same_ids_list, old_ids, new_ids

# ------------------------------ #
#         MAIN FUNCTION          #
# ------------------------------ #

def main():
    """
    1) Check if there's a latest 'spread_subset_test_{n}.jsonl' file.
       - If no, create new file from the first 20 valid articles, do no comparisons.
       - If yes, read its IDs, create a new file, reprocess those articles (with two prompts),
         and compare results with all older versions.
    """

    latest_file = find_latest_output_file(base_prefix="spread_subset_test")
    if not latest_file:
        # No existing file --> create spread_subset_test_2.jsonl from the first 20 valid articles
        new_file = get_next_output_file(base_prefix="spread_subset_test")
        print(f"No existing 'spread_subset_test_n.jsonl' found.\nCreating new file: {new_file}")
        
        # Read first 20 valid articles
        valid_articles = []
        count_found = 0

        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fulltext = article.get("fulltext", "")
                    if fulltext.strip():
                        article = assign_id_if_missing(article)
                        valid_articles.append(article)
                        count_found += 1
                        if count_found == 20:
                            break
        except FileNotFoundError:
            print(f"Input file '{INPUT_FILE}' not found.")
            return

        if not valid_articles:
            print("No valid articles found in the input file.")
            return

        # Process them with the TWO PROMPTS
        processed_articles = []
        print("Processing first 20 valid articles from scratch...")
        for article in tqdm(valid_articles, desc="Processing articles"):
            fulltext = article.get("fulltext", "")
            result = process_article_two_prompts(fulltext)
            article["spread_risk"] = result["spread_risk"]
            article["spread_feedback"] = result["feedback"]
            processed_articles.append(article)

        # Write out the new file
        with open(new_file, 'w', encoding='utf-8') as nf:
            for rec in processed_articles:
                nf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nCreated {new_file} with {len(processed_articles)} articles processed.")
        print("No comparisons performed, as this is the first file.")
        return

    # If we get here, we have an existing file
    print(f"Latest existing file: {latest_file}")

    # Read all IDs from that file
    last_ids = []
    try:
        with open(latest_file, 'r', encoding='utf-8') as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    aid = record.get("id")
                    if aid:
                        last_ids.append(aid)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line in the latest file.")
    except FileNotFoundError:
        print(f"File not found: {latest_file}")
        return

    if not last_ids:
        print(f"No IDs found in {latest_file}. Nothing to process.")
        return

    print(f"Found {len(last_ids)} IDs in {latest_file}.")

    # We create the next version file
    new_file = get_next_output_file(base_prefix="spread_subset_test")
    print(f"Creating new file: {new_file}")

    # Collect articles by ID from distribution_mortality_subset.jsonl
    last_ids_set = set(last_ids)
    articles_dict = {}
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article = assign_id_if_missing(article)
                aid = article["id"]
                if aid in last_ids_set:
                    articles_dict[aid] = article
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        return

    missing_ids = last_ids_set - set(articles_dict.keys())
    if missing_ids:
        print("WARNING: Some IDs from the latest file were not found in distribution_mortality_subset.jsonl:")
        for mid in missing_ids:
            print("  ", mid)

    processed_articles = []
    not_found_count = 0

    print("Processing articles from distribution_mortality_subset.jsonl that match existing IDs...")
    for aid in tqdm(last_ids, desc="Processing articles"):
        if aid not in articles_dict:
            not_found_count += 1
            continue
        article = articles_dict[aid]
        fulltext = article.get("fulltext", "")
        # Now we do the TWO prompt calls
        result = process_article_two_prompts(fulltext)
        article["spread_risk"] = result["spread_risk"]
        article["spread_feedback"] = result["feedback"]
        processed_articles.append(article)

    # Write them to the new file
    with open(new_file, 'w', encoding='utf-8') as nf:
        for rec in processed_articles:
            nf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nCreated {new_file} with {len(processed_articles)} articles processed.")
    if not_found_count > 0:
        print(f"Note: {not_found_count} articles were missing from {INPUT_FILE} and were not processed.")

    # Compare the new file to all older versions
    pattern = re.compile(r'^spread_subset_test_(\d+)\.jsonl$')
    files_to_compare = []
    for fname in os.listdir('.'):
        if fname == new_file:
            continue
        match = pattern.match(fname)
        if match:
            files_to_compare.append(fname)

    # Sort them by their numeric suffix
    files_to_compare.sort(key=lambda x: int(pattern.match(x).group(1)))

    for oldf in files_to_compare:
        print(f"\nComparing new file ({new_file}) to older file ({oldf})...")
        diffs, same_ids, old_ids, new_ids = compare_spread_fields(oldf, new_file)

        if diffs:
            print(f"  Differences in spread_risk for these IDs: {', '.join(diffs)}")
        else:
            print("  No differences in spread_risk.")

        if same_ids:
            print(f"  These IDs had the same 'spread_risk': {', '.join(same_ids)}")
        else:
            print("  No IDs matched or had same spread_risk with this old file.")

        print(f"  # of IDs in {oldf}: {len(old_ids)}; # of IDs in {new_file}: {len(new_ids)}")


if __name__ == "__main__":
    main()
