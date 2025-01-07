import json
import os
import sys
import uuid
import logging
import builtins
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, OpenAIError

# Configure logging
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

INPUT_FILE = 'distribution_set_final_deepseek.jsonl'
BASE_OUTPUT_FILENAME = 'transmission_quantitative_deepseek.jsonl'
DISCARDED_OUTPUT_FILENAME = 'transmission_quantitative_discarded_deepseek.jsonl'

MODEL = 'deepseek-chat'  # Updated to use the "deepseek-chat" model

# Import NLTK and download necessary resources
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

def append_jsonl(record, filename):
    with open(filename, 'a', encoding='utf-8') as nf:
        nf.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_existing_output(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError:
                    logging.warning("Skipping invalid JSON line in output file.")
    except FileNotFoundError:
        pass
    return data

def assign_id_if_missing(article, default_prefix="article_"):
    if "id" not in article:
        article["id"] = default_prefix + str(uuid.uuid4())
    return article

def verify_extracted_statements(article_id, fulltext, extracted_data):
    sents = sent_tokenize(fulltext)
    discarded = []
    keys_to_check = [k for k in extracted_data if k.startswith("affirm_") or k.startswith("deny_")]
    for key in keys_to_check:
        stmt = extracted_data.get(key, "")
        if isinstance(stmt, str) and not any(sent.strip() == stmt.strip() for sent in sents):
            discarded.append((key, stmt))
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

def make_api_call(prompt, article_id=None):
    try:
        logging.info(f"Sending API request for article {article_id} with prompt:\n{prompt}")
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
        reply = response.choices[0].message.content.strip()
        logging.info(f"Raw API response for article {article_id}:\n{reply}")
        if reply.startswith("```") and reply.endswith("```"):
            reply = "\n".join(reply.split("\n")[1:-1]).strip()
        return json.loads(reply)
    except OpenAIError as oae:
        logging.error(f"OpenAI API error for article {article_id}: {oae}", exc_info=True)
        return {"error": True}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON response for article {article_id}: {reply}")
        return {"error": True}
    except Exception as e:
        logging.error(f"Unexpected error for article {article_id}: {e}", exc_info=True)
        return {"error": True}

def process_article_six_prompts(article_id, fulltext):
    if not fulltext.strip():
        logging.warning(f"Article {article_id} has empty fulltext. Skipping.")
        return {"error": True}, None

    # Define new prompts explicitly
    deny_keywords_prompt = f"""
    You are an extremely thorough analytical assistant.
    
    Step 1: Systematically check the article text below for any statements that contain the words 'pandemic', 'pandemics', 'coronavirus', 'Covid', or 'COVID-19', while DENYING similarities to bird flu or asserting that bird flu might not become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "deny_keywords_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "deny_keywords_statement_1": "...",
      "deny_keywords_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    deny_why_prompt = f"""
    You are an extremely thorough analytical assistant.
    
    Step 1: Systematically check the article text below for any statements PROVIDE SCIENTIFIC REASONS WHY bird flu DOES NOT have the potential to, in the future, evolve or otherwise become more transmissible, infectious, or start to spread or transmit from human to human, or become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "deny_why_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "deny_why_statement_1": "...",
      "deny_why_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    deny_transmissible_prompt = f"""
    You are an extremely thorough analytical assistant.
    
    Step 1: Systematically check the article text below for any statements that suggest, imply, or explicitly assert that the bird flu virus being discussed DOES NOT have the potential to, in the future, evolve or otherwise become more transmissible, infectious, or start to spread or transmit from human to human, or become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "deny_transmissible_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "deny_transmissible_statement_1": "...",
      "deny_transmissible_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    affirm_keywords_prompt = f"""
    You are an extremely thorough analytical assistant.
    
    Step 1: Systematically check the article text below for any statements that contain the words 'pandemic', 'pandemics', 'coronavirus', 'Covid', or 'COVID-19', without denying similarities to bird flu or asserting that bird flu might not become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "affirm_keywords_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "affirm_keywords_statement_1": "...",
      "affirm_keywords_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    affirm_why_prompt = f"""
    You are an extremely thorough analytical assistant.
    
    Step 1: Systematically check the article text below for any statements PROVIDE SCIENTIFIC REASONS WHY bird flu has the potential to, in the future, evolve or otherwise become more transmissible, infectious, or start to spread or transmit from human to human, or become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "affirm_why_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "affirm_why_statement_1": "...",
      "affirm_why_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    affirm_transmissible_prompt = f"""
    You are an extremely thorough analytical assistant.
        
    Step 1: Systematically check the article text below for any statements that suggest, imply, or explicitly assert that the bird flu virus being discussed has the potential to, in the future, evolve or otherwise become more transmissible, infectious, or start to spread or transmit from human to human, or become a pandemic.
    
    Step 2: For EACH such statement you find, return it in a unique "affirm_transmissible_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "affirm_transmissible_statement_1": "...",
      "affirm_transmissible_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

    # Make API calls in reverse order
    deny_keywords_result = make_api_call(deny_keywords_prompt, article_id=article_id)
    deny_why_result = make_api_call(deny_why_prompt, article_id=article_id)
    deny_transmissible_result = make_api_call(deny_transmissible_prompt, article_id=article_id)
    affirm_keywords_result = make_api_call(affirm_keywords_prompt, article_id=article_id)
    affirm_why_result = make_api_call(affirm_why_prompt, article_id=article_id)
    affirm_transmissible_result = make_api_call(affirm_transmissible_prompt, article_id=article_id)

    # Initialize final_data
    final_data = {}

    # Process affirm_transmissible and affirm_why
    affirm_transmissible_statements = {k: v for k, v in affirm_transmissible_result.items() if k.startswith("affirm_transmissible_statement_")}
    affirm_why_statements = {k: v for k, v in affirm_why_result.items() if k.startswith("affirm_why_statement_")}

    # Remove duplicates from affirm_transmissible_statements
    unique_affirm_transmissible = {}
    for key, stmt in affirm_transmissible_statements.items():
        if stmt not in affirm_why_statements.values():
            unique_affirm_transmissible[key] = stmt
    final_data.update(unique_affirm_transmissible)
    final_data["affirm_transmissible_count"] = len(unique_affirm_transmissible)

    # Add affirm_why_statements
    final_data.update(affirm_why_statements)
    final_data["affirm_why_count"] = len(affirm_why_statements)

    # Process affirm_keywords
    affirm_keywords_statements = {k: v for k, v in affirm_keywords_result.items() if k.startswith("affirm_keywords_statement_")}
    existing_affirm_statements = list(unique_affirm_transmissible.values()) + list(affirm_why_statements.values())
    unique_affirm_keywords = {}
    for key, stmt in affirm_keywords_statements.items():
        if stmt not in existing_affirm_statements:
            unique_affirm_keywords[key] = stmt
    final_data.update(unique_affirm_keywords)
    final_data["affirm_keywords_count"] = len(unique_affirm_keywords)

    # Process deny_transmissible and deny_why
    deny_transmissible_statements = {k: v for k, v in deny_transmissible_result.items() if k.startswith("deny_transmissible_statement_")}
    deny_why_statements = {k: v for k, v in deny_why_result.items() if k.startswith("deny_why_statement_")}

    # Remove duplicates from deny_transmissible_statements
    unique_deny_transmissible = {}
    for key, stmt in deny_transmissible_statements.items():
        if stmt not in deny_why_statements.values():
            unique_deny_transmissible[key] = stmt
    final_data.update(unique_deny_transmissible)
    final_data["deny_transmissible_count"] = len(unique_deny_transmissible)

    # Add deny_why_statements
    final_data.update(deny_why_statements)
    final_data["deny_why_count"] = len(deny_why_statements)

    # Process deny_keywords
    deny_keywords_statements = {k: v for k, v in deny_keywords_result.items() if k.startswith("deny_keywords_statement_")}
    existing_deny_statements = list(unique_deny_transmissible.values()) + list(deny_why_statements.values())
    unique_deny_keywords = {}
    for key, stmt in deny_keywords_statements.items():
        if stmt not in existing_deny_statements:
            unique_deny_keywords[key] = stmt
    final_data.update(unique_deny_keywords)
    final_data["deny_keywords_count"] = len(unique_deny_keywords)

    # Verify extracted statements and handle discarded ones
    discard_info = verify_extracted_statements(article_id, fulltext, final_data)
    return final_data, discard_info

def get_next_output_file(base_name='distribution_subset_transmission'):
    n = 1
    while True:
        candidate = f"{base_name}_{n}.jsonl"
        if not os.path.exists(candidate):
            return candidate
        n += 1

def main():
    output_filename = BASE_OUTPUT_FILENAME
    existing_output_data = []
    existing_dict = {}

    if os.path.isfile(output_filename):
        print(f"Found existing {output_filename}.")
        choice = input("Would you like to continue coding in this file? [yes/no] ").strip().lower()
        if choice.startswith('y'):
            existing_output_data = load_existing_output(output_filename)
            existing_dict = {a["id"]: a for a in existing_output_data}
        else:
            choice2 = input("Create a new output file? [yes/no] ").strip().lower()
            if choice2.startswith('y'):
                output_filename = get_next_output_file('distribution_subset_transmission')
                print(f"Using new output file: {output_filename}")
            else:
                sys.exit("Terminating script at user request.")
    else:
        print(f"No existing {output_filename}, starting fresh.")

    # Load all articles
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
    
    # Sort articles in reverse order based on ID
    all_articles.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else float('inf'), reverse=True)

    # Open the output file in the appropriate mode
    if os.path.isfile(output_filename) and choice.startswith('y'):
        mode = 'a'
    else:
        mode = 'w'

    # Process each article one by one
    discard_list = []
    processed_articles = []  # Store processed articles for reverse order writing
    
    for i, art in enumerate(tqdm(all_articles, desc="Coding Articles")):
        aid = art["id"]
        if aid in existing_dict:
            logging.info(f"Article {aid} already processed. Skipping.")
            continue
        if "fulltext" not in art:
            art["processing_error"] = "Missing fulltext."
            processed_articles.append(art)
            continue

        fulltext = art["fulltext"]
        final_data, discard_info = process_article_six_prompts(aid, fulltext)
        if "error" in final_data:
            logging.error(f"Error processing article {aid}.")
            continue

        # Update article with final_data
        art.update(final_data)
        processed_articles.append(art)

        # Handle discard_info
        if discard_info:
            discard_list.append(discard_info)

    # Write processed articles to output file
    with open(output_filename, mode, encoding='utf-8') as outf:
        for article in processed_articles:
            outf.write(json.dumps(article, ensure_ascii=False) + "\n")

    # Save discarded statements to DISCARDED_OUTPUT_FILENAME
    if discard_list:
        with open(DISCARDED_OUTPUT_FILENAME, 'w', encoding='utf-8') as df:
            for discard in discard_list:
                df.write(json.dumps(discard, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
