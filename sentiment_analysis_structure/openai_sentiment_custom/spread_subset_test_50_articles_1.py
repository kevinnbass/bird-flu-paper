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

# Input data
INPUT_FILE = 'distribution_mortality_subset.jsonl'
OLD_FILE = 'spread_subset_test_3.jsonl'
# The new file we generate
NEW_BASE_PREFIX = 'spread_subset_test_50_articles'

MODEL = 'gpt-4o'  # No retries, using 'gpt-4o'.

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

def find_latest_output_file(base_prefix):
    """
    Find all files named '{base_prefix}_{n}.jsonl' in the current directory.
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

def get_next_output_file(base_prefix):
    """
    Starting from suffix 1, find the first '{base_prefix}_{n}.jsonl' that doesn't exist.
    If 1 exists, try 2, etc.
    """
    n = 1
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
        logging.error("Detailed error during API call:", exc_info=True)
        return {
            "spread_risk": "error",
            "feedback": "API call failed or invalid response."
        }

def process_article_two_prompts(fulltext):
    """
    Runs TWO prompts on the same article text.

    - Prompt #1: PROMPT_SPREAD_RISK_1 (affirms risk).
    - Prompt #2: PROMPT_SPREAD_RISK_2 (denies/minimizes risk).

    If either call returns "yes", final 'spread_risk' is "yes". 
    Otherwise, "no".

    The final 'feedback' is:
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

    # If either is "yes", overall is "yes"
    if spread1 == "yes" or spread2 == "yes":
        final_spread = "yes"
    else:
        final_spread = "no"

    final_feedback = f"AFFIRMS RISK: {fb1} DENIES RISK: {fb2}"
    return {
        "spread_risk": final_spread,
        "feedback": final_feedback
    }

def load_ids_from_file(filename):
    """
    Load all 'id' fields from the specified filename (JSONL). Return as a set.
    """
    ids_set = set()
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    aid = record.get("id")
                    if aid:
                        ids_set.add(aid)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {filename}")
    return ids_set

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
                    logging.warning(f"Skipping invalid JSON in old file {old_file}")

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
                    logging.warning(f"Skipping invalid JSON in new file {new_file}")

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
#            Main App            #
# ------------------------------ #

def main():
    """
    1) We generate a new file named spread_subset_test_50_articles_{n}.jsonl 
       (starting from n=1, so spread_subset_test_50_articles_1.jsonl).
    2) We read the first 50 new unique articles (i.e. with IDs not in spread_subset_test_3.jsonl).
    3) We process them with the two prompts approach.
    4) We write them to that new file.
    5) Then we append the 20 old articles from spread_subset_test_3.jsonl at the end.
    6) We do versioning logic: if there is a prior file named spread_subset_test_50_articles_{m}, 
       we compare spread_risk fields. 
       But if n=1, there's no older file to compare, so no comparisons occur. 
       On subsequent runs (n=2, etc.), comparisons happen.
    """

    # Step 1: determine the next version file name
    # e.g. "spread_subset_test_50_articles_1.jsonl" if none exist yet
    new_file = get_next_output_file(NEW_BASE_PREFIX)
    print(f"New output file will be: {new_file}")

    # Step 2: read the old file's IDs (spread_subset_test_3.jsonl)
    old_ids_set = load_ids_from_file(OLD_FILE)
    print(f"Loaded {len(old_ids_set)} IDs from {OLD_FILE} to skip...")

    # Step 3: gather 50 new unique articles from distribution_mortality_subset.jsonl
    new_articles = []
    count_found = 0
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                if count_found == 50:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    continue

                article = assign_id_if_missing(article)
                aid = article["id"]
                if aid in old_ids_set:
                    # skip articles with the same ID as in spread_subset_test_3.jsonl
                    continue

                fulltext = article.get("fulltext", "")
                if fulltext.strip():
                    new_articles.append(article)
                    count_found += 1
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        return

    if count_found < 50:
        print(f"WARNING: we only found {count_found} new unique articles. We wanted 50, but data may be insufficient.")

    if not new_articles:
        print("No new articles found. Exiting.")
        return

    # Step 4: process them with the two prompts approach
    processed_articles = []
    print(f"Processing {len(new_articles)} new articles with two prompts...")
    for article in tqdm(new_articles, desc="Processing new articles"):
        fulltext = article.get("fulltext", "")
        result = process_article_two_prompts(fulltext)
        article["spread_risk"] = result["spread_risk"]
        article["spread_feedback"] = result["feedback"]
        processed_articles.append(article)

    # Step 5: write them to the new file
    with open(new_file, 'w', encoding='utf-8') as nf:
        for rec in processed_articles:
            nf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(processed_articles)} new articles to {new_file}.")

    # Step 6: append the 20 old articles from spread_subset_test_3.jsonl
    appended_count = 0
    if os.path.isfile(OLD_FILE):
        with open(OLD_FILE, 'r', encoding='utf-8') as oldf, \
             open(new_file, 'a', encoding='utf-8') as nf:
            for line in oldf:
                line_stripped = line.strip()
                if line_stripped:
                    nf.write(line_stripped + "\n")
                    appended_count += 1
        print(f"Appended {appended_count} lines from {OLD_FILE} onto the end of {new_file}.")
    else:
        print(f"Could not find {OLD_FILE}, so not appending anything.")

    total_lines = len(processed_articles) + appended_count
    print(f"Final file {new_file} now has {total_lines} total lines.")

    # ------------------------------ #
    # Versioning & Comparisons step  #
    # ------------------------------ #
    # If this is the "first" file, there's no older version to compare to.
    # But if there's an older file (like spread_subset_test_50_articles_{n-1}.jsonl),
    # we do the standard comparison.

    # Find the "latest" file among this same prefix that isn't new_file
    # Because we might have multiple versions: spread_subset_test_50_articles_1, 2, 3, ...
    existing_latest = find_latest_output_file(NEW_BASE_PREFIX)
    # existing_latest might be new_file itself if it's the only one

    # We'll gather all older files, ignoring new_file
    pattern = re.compile(rf'^{NEW_BASE_PREFIX}_(\d+)\.jsonl$')
    files_to_compare = []
    for fname in os.listdir('.'):
        if fname == new_file:
            continue
        match = pattern.match(fname)
        if match:
            files_to_compare.append(fname)

    # sort them by numeric suffix
    files_to_compare.sort(key=lambda x: int(pattern.match(x).group(1)))

    if files_to_compare:
        # We have older versions to compare
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
    else:
        # No older versions found for comparison
        print("No older versions found to compare against. This must be the first iteration.")


if __name__ == "__main__":
    main()
