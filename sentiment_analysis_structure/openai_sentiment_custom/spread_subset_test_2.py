import json
import openai
import os
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
from openai.error import OpenAIError
import logging

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

# OLD file we compare against
OLD_OUTPUT_FILE = 'spread_subset_test.jsonl'

# NEW file we generate
NEW_OUTPUT_FILE = 'spread_subset_test_2.jsonl'

MODEL = 'gpt-4o'

# ------------------------------ #
#         Single Prompt          #
# ------------------------------ #
PROMPT_SPREAD_RISK = """
You are an analytical assistant. Your task is to see if the text explicitly or strongly implies that bird flu has potential for:

1) Spreading human to human (i.e., actual or likely human-to-human transmission),
2) Causing or being on the brink of causing a large-scale pandemic in humans,
3) Mutating or becoming more infectious in a manner that clearly threatens large-scale human outbreaks.

IMPORTANT clarifications:
- If the article ONLY mentions:
  - That bird flu can infect humans,
  - That bird flu can be lethal in humans,
  - Historical or current human infections or fatalities,
  - Or that humans can catch it from contact with infected birds,
  then DO NOT code as "yes".

- You should code "yes" ONLY if the text states or strongly implies:
  - Human-to-human transmission already happening or is very likely,
  - A serious or imminent risk of a human pandemic,
  - The virus is mutating or has mutated to spread efficiently among humans, or
  - A credible warning that it could soon start large-scale human outbreaks.

- However, if the article text contains the word "pandemic" anywhere, override the rules above and code "spread_risk" = "yes".

Additionally, for the "feedback" field, provide TWO things:
1) The reasoning for your determination,
2) All exact quotations from the article text (verbatim) that led to your determination.

Respond ONLY with a JSON object in the following form:

{{
  "spread_risk": "yes" or "no",
  "feedback": "Your reasoning + verbatim quotations here"
}}

Article Text:
{{fulltext}}
"""

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def assign_id_if_missing(article, default_prefix="article_"):
    """Adds a unique ID to the article if it doesn't already have one."""
    if "id" not in article:
        article["id"] = default_prefix + str(uuid.uuid4())
    return article

def make_api_call(prompt_template, fulltext):
    """
    Makes a single API call with NO retries.
    Returns a dict with "spread_risk" and "feedback".
    """
    # Format the prompt with the article text
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
            raise ValueError("Response is not a valid JSON object. Response was:\n" + reply)
        
        result = json.loads(reply)
        return result

    except (OpenAIError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Error during API call: {e}")
        return {
            "spread_risk": "error",
            "feedback": "API call failed or invalid response."
        }

def process_article(fulltext):
    """
    Analyzes the article's fulltext using the single prompt and returns a dict
    with spread_risk and feedback. If fulltext is missing/empty, returns "error".
    """
    if not fulltext.strip():
        return {
            "spread_risk": "error",
            "feedback": "No text provided."
        }
    result = make_api_call(PROMPT_SPREAD_RISK, fulltext)
    spread_risk = result.get("spread_risk", "error")
    feedback = result.get("feedback", "No feedback")
    return {
        "spread_risk": spread_risk,
        "feedback": feedback
    }

def compare_spread_fields(old_file, new_file):
    """
    Compares the 'spread_risk' field in the old_file vs. new_file by matching 'id'.
    Returns two lists:
        diffs: list of IDs whose 'spread_risk' differ
        same_ids: list of IDs whose 'spread_risk' is the same
    Also returns sets of all old IDs and all new IDs for debugging.
    """
    old_data = {}
    new_data = {}

    # Read the old file
    old_ids = set()
    if os.path.isfile(old_file):
        with open(old_file, 'r', encoding='utf-8') as f_old:
            for line in f_old:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "id" in record:
                        old_data[record["id"]] = record
                        old_ids.add(record["id"])
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in old file.")
    else:
        logging.warning(f"Old file '{old_file}' not found. Comparison will yield no results.")
        return [], [], set(), set()

    # Read the new file
    new_ids = set()
    if os.path.isfile(new_file):
        with open(new_file, 'r', encoding='utf-8') as f_new:
            for line in f_new:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "id" in record:
                        new_data[record["id"]] = record
                        new_ids.add(record["id"])
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in new file.")
    else:
        logging.warning(f"New file '{new_file}' not found. Comparison will yield no results.")
        return [], [], old_ids, new_ids

    # Compare spread_risk field
    diffs = []
    same_ids = []
    for article_id, new_record in new_data.items():
        if article_id in old_data:
            old_spread = old_data[article_id].get("spread_risk")
            new_spread = new_record.get("spread_risk")
            if old_spread != new_spread:
                diffs.append(article_id)
            else:
                same_ids.append(article_id)

    return diffs, same_ids, old_ids, new_ids

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    1) Reads from INPUT_FILE until we find 20 articles that have a non-empty 'fulltext' field.
    2) Processes those 20 articles, skipping those whose IDs are already processed,
       and appends results to NEW_OUTPUT_FILE (spread_subset_test_2.jsonl).
    3) After processing, compares the new file to the old file (spread_subset_test.jsonl)
       and prints any IDs where spread_risk differs, and also prints any IDs
       whose spread_risk are the same.
    4) Finally, prints all IDs found in each file for debugging.
    """
    # Create the new output file if it doesn't exist
    if not os.path.isfile(NEW_OUTPUT_FILE):
        with open(NEW_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass

    # Build a set of processed IDs from the existing new output
    processed_ids = set()
    try:
        with open(NEW_OUTPUT_FILE, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_entry = json.loads(line)
                    if "id" in existing_entry:
                        processed_ids.add(existing_entry["id"])
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in existing new output.")
    except FileNotFoundError:
        logging.info(f"No existing file named '{NEW_OUTPUT_FILE}' found. A new one will be created.")

    # Grab the first 20 valid articles (those that parse and have non-empty 'fulltext')
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
                    continue  # skip invalid JSON lines

                fulltext = article.get("fulltext", "")
                if fulltext.strip():
                    valid_articles.append(article)
                    count_found += 1
                    if count_found == 20:
                        break
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        logging.error(f"Input file '{INPUT_FILE}' not found.")
        return

    # If we didn't find any valid articles, just quit
    if not valid_articles:
        print("No valid articles found with non-empty fulltext.")
        return

    # Process these valid articles, skipping any that are already processed in the new file
    with open(NEW_OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
        for idx, article in enumerate(tqdm(valid_articles, desc="Processing 20 valid articles")):
            article = assign_id_if_missing(article)

            if article["id"] in processed_ids:
                continue

            fulltext = article.get("fulltext", "")
            result = process_article(fulltext)
            article["spread_risk"] = result["spread_risk"]
            article["spread_feedback"] = result["feedback"]

            outfile.write(json.dumps(article, ensure_ascii=False) + "\n")
            outfile.flush()
            processed_ids.add(article["id"])

    print(f"Processing completed. Created/updated: {NEW_OUTPUT_FILE}")

    # Compare new file to old file
    differences, same_ids, old_ids, new_ids = compare_spread_fields(OLD_OUTPUT_FILE, NEW_OUTPUT_FILE)

    if differences:
        print("\nDifferences in spread_risk found for the following IDs:")
        for diff_id in differences:
            print(diff_id)
    else:
        print("\nNo differences in spread_risk field found between the two files.")

    if same_ids:
        print("\nThese IDs had the same 'spread_risk' in both files:")
        for s_id in same_ids:
            print(s_id)
    else:
        print("\nNo IDs matched between the files to compare (same 'spread_risk').")

    # Debug: print all IDs in both files
    print("\nALL IDs in OLD FILE (spread_subset_test.jsonl):")
    for oid in sorted(old_ids):
        print("  ", oid)

    print("\nALL IDs in NEW FILE (spread_subset_test_2.jsonl):")
    for nid in sorted(new_ids):
        print("  ", nid)


if __name__ == "__main__":
    main()
