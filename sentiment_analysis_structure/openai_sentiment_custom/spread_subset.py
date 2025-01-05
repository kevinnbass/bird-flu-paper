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
OUTPUT_FILE = 'spread_subset_test.jsonl'

# Use "gpt-4o" per your request
MODEL = 'gpt-4o'

# ------------------------------ #
#         Single Prompt          #
# ------------------------------ #

# Double-curly braces around literal JSON braces so str.format() doesn't treat them as placeholders
PROMPT_SPREAD_RISK = """
You are an analytical assistant. Given the article text below, determine if it suggests or implies that bird flu has potential to:
- Spread to humans,
- Cause a pandemic,
- Become more infectious,
- Or start spreading human to human.

If any of these are implied, then "spread_risk" = "yes".
Otherwise, "spread_risk" = "no".

Additionally, for the "feedback" field, provide TWO things:
1) The reasoning for your determination.
2) All exact quotations from the article text (verbatim) that led to your determination.

Respond ONLY with a JSON object in the following form:

{{
  "spread_risk": "yes" or "no",
  "feedback": "Your reasoning + verbatim quotations here"
}}

Article Text:
{fulltext}
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
    If the call fails, returns {"spread_risk": "error", "feedback": "API call failed ..."}.
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
    Analyzes the article fulltext using the single prompt and returns spread_risk, feedback.
    If fulltext is missing/empty, set spread_risk = "error".
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

# ------------------------------ #
#            Main App            #
# ------------------------------ #

def main():
    """
    Reads from INPUT_FILE until we find 20 articles that have a non-empty 'fulltext' field.
    Then processes those 20 articles, skipping IDs that are already processed.
    Appends results to OUTPUT_FILE.
    """
    # Create the output file if it doesn't exist
    if not os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            pass
    
    # Build a set of processed IDs from the existing output
    processed_ids = set()
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_entry = json.loads(line)
                    if "id" in existing_entry:
                        processed_ids.add(existing_entry["id"])
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON in existing output.")
    except FileNotFoundError:
        logging.info("No existing output file found. A new one will be created.")
    
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

                # Check if article has non-empty "fulltext"
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
    
    # Now we have up to 20 articles with a non-empty fulltext
    if not valid_articles:
        print("No valid articles found with non-empty fulltext.")
        return
    
    # Process these valid articles
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
        for idx, article in enumerate(tqdm(valid_articles, desc="Processing 20 valid articles")):
            article = assign_id_if_missing(article)
            
            # Skip if processed
            if article["id"] in processed_ids:
                continue
            
            fulltext = article.get("fulltext", "")
            result = process_article(fulltext)
            article["spread_risk"] = result["spread_risk"]
            article["spread_feedback"] = result["feedback"]
            
            outfile.write(json.dumps(article) + "\n")
            outfile.flush()
            processed_ids.add(article["id"])
    
    print("Processing completed. Created/updated:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
