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

# Function to create prompts dynamically
def create_prompt(category, keyword, instruction, fulltext, deny=False):
    if deny:
        content = f"explicitly downplay, minimize, or otherwise suggest that the risk of bird flu is low for: {instruction}."
    else:
        content = f"suggest, imply, or explicitly state that bird flu has potential to: {instruction}."
    return f"""
    You are an analytical assistant.
    
    Step 1: Systematically check the article text below for any statements that {content}
    
    Step 2: For EACH such statement you find, return it in a unique "{category}_{keyword}_statement_n" field (verbatim text, no summarization).
    
    Respond ONLY with a JSON object in the format:
    
    {{
      "{category}_{keyword}_statement_1": "...",
      "{category}_{keyword}_statement_2": "...",
      ...
    }}
    
    Article Text: {fulltext}
    """

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
    affirm_keys = sorted(k for k in extracted_data if k.startswith("affirm_"))
    valid_affirms = []
    for key in affirm_keys:
        st = extracted_data[key]
        if isinstance(st, str) and any(sent.strip() == st.strip() for sent in sents):
            valid_affirms.append((key, st))
        else:
            discarded.append((key, st))
    for k in affirm_keys:
        del extracted_data[k]
    for i, (orig_key, st) in enumerate(valid_affirms, start=1):
        extracted_data[orig_key] = st  # Preserve the original key
    extracted_data["affirm_count"] = len(valid_affirms)

    deny_keys = sorted(k for k in extracted_data if k.startswith("deny_"))
    valid_denies = []
    for key in deny_keys:
        st = extracted_data[key]
        if isinstance(st, str) and any(sent.strip() == st.strip() for sent in sents):
            valid_denies.append((key, st))
        else:
            discarded.append((key, st))
    for k in deny_keys:
        del extracted_data[k]
    for i, (orig_key, st) in enumerate(valid_denies, start=1):
        extracted_data[orig_key] = st  # Preserve the original key
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

def process_article_ten_prompts(article_id, fulltext):
    if not fulltext.strip():
        logging.warning(f"Article {article_id} has empty fulltext. Skipping.")
        return {"error": True}, None

    affirm_prompts = [
        create_prompt("affirm", "crisis", "Cause a public health crisis.", fulltext),
        create_prompt("affirm", "evolve", "Evolve to become more transmissible or infectious to humans in the future.", fulltext),
        create_prompt("affirm", "pandemic", "Evolve to become more transmissible or infectious to humans in the future.", fulltext),
        create_prompt("affirm", "humanspread", "Evolve to spread human to human.", fulltext),
        create_prompt("affirm", "keyword", "Include the word 'pandemic', 'coronavirus', 'Covid', or 'COVID-19' and that ***DO NOT DENY*** that bird flu might be similar.", fulltext)
    ]

    deny_prompts = [
        create_prompt("deny", "crisis", "Causing a public health crisis.", fulltext, deny=True),
        create_prompt("deny", "evolve", "Evolving to become more transmissible or infectious to humans.", fulltext, deny=True),
        create_prompt("deny", "pandemic", "Causing a pandemic.", fulltext, deny=True),
        create_prompt("deny", "humanspread", "Starting to spread human to human.", fulltext, deny=True),
        create_prompt("deny", "keyword", "using the word 'pandemic', 'coronavirus', 'Covid', or 'COVID-19' that ***DENIES*** that bird flu might be similar.", fulltext, deny=True)
    ]

    affirm_results = []
    for prompt in affirm_prompts:
        result = make_api_call(prompt, article_id=article_id)
        if "error" in result:
            logging.error(f"Error processing affirm prompt for article {article_id}.")
            return {"error": True}, None
        affirm_results.append(result)

    deny_results = []
    for prompt in deny_prompts:
        result = make_api_call(prompt, article_id=article_id)
        if "error" in result:
            logging.error(f"Error processing deny prompt for article {article_id}.")
            return {"error": True}, None
        deny_results.append(result)

    final_data = {}
    affirm_count = 0
    for result in affirm_results:
        for k, v in result.items():
            if k.startswith("affirm_"):
                final_data[k] = v
                affirm_count += 1
    final_data["affirm_count"] = affirm_count

    deny_count = 0
    for result in deny_results:
        for k, v in result.items():
            if k.startswith("deny_"):
                final_data[k] = v
                deny_count += 1
    final_data["deny_count"] = deny_count

    discard_info = verify_extracted_statements(article_id, fulltext, final_data)
    return final_data, discard_info

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

    # Open the output file in the appropriate mode
    if os.path.isfile(output_filename) and choice.startswith('y'):
        mode = 'a'
    else:
        mode = 'w'

    # Process each article one by one
    discard_list = []
    with open(output_filename, mode, encoding='utf-8') as outf:
        for i, art in enumerate(tqdm(all_articles, desc="Coding Articles")):
            aid = art["id"]
            if aid in existing_dict:
                logging.info(f"Article {aid} already processed. Skipping.")
                continue
            if "fulltext" not in art:
                art["processing_error"] = "Missing fulltext"
                logging.warning(f"Article {aid} is missing 'fulltext' key. Adding processing error.")
                append_jsonl(art, output_filename)
                continue
            fulltext = art["fulltext"]
            result, discard_info = process_article_ten_prompts(aid, fulltext)
            if "error" in result:
                art["api_error"] = True
                append_jsonl(art, output_filename)
                continue
            # Merge the result into the article
            for k, v in result.items():
                art[k] = v
            # Write the processed article immediately
            append_jsonl(art, output_filename)
            # Handle discarded statements
            if discard_info is not None:
                append_jsonl(discard_info, DISCARDED_OUTPUT_FILENAME)

    print(f"\nAll done! Output file: {output_filename}")
    if os.path.isfile(DISCARDED_OUTPUT_FILENAME):
        print(f"Discarded statements written to {DISCARDED_OUTPUT_FILENAME}.")
    else:
        print("No discarded statements were found.")

if __name__ == "__main__":
    main()
