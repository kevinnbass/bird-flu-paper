import json
import logging
import sys

# Configure logging to output to terminal (stdout)
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

sentiment_analysis_file = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'
examine_complete_file = 'processed_all_articles_examine_complete.json'
new_examine_file = 'processed_all_articles_examine_complete_with_huffpost.json'

logging.info("Starting process to append HuffPost entries.")

# Extract HuffPost articles from the sentiment analysis file
huffpost_entries = []
logging.info(f"Reading from {sentiment_analysis_file} to find HuffPost entries.")
try:
    with open(sentiment_analysis_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            if article.get("media_outlet") == "HuffPost":
                huffpost_entries.append(article)
    logging.info(f"Found {len(huffpost_entries)} HuffPost entries.")
except Exception as e:
    logging.error(f"Failed to read from {sentiment_analysis_file}: {e}")
    raise e

# Load the existing examine_complete file
logging.info(f"Loading data from {examine_complete_file}.")
try:
    with open(examine_complete_file, 'r', encoding='utf-8') as f:
        examine_data = json.load(f)
    if not isinstance(examine_data, list):
        raise ValueError(f"{examine_complete_file} should contain a JSON array at the top level.")
    logging.info(f"Loaded {len(examine_data)} entries from {examine_complete_file}.")
except Exception as e:
    logging.error(f"Failed to load data from {examine_complete_file}: {e}")
    raise e

# Append HuffPost entries to the examine_complete data
initial_count = len(examine_data)
examine_data.extend(huffpost_entries)
logging.info(f"Appended HuffPost entries. Total entries now: {len(examine_data)} (was {initial_count}).")

# Write back the updated data to a new file
logging.info(f"Writing combined data to {new_examine_file}.")
try:
    with open(new_examine_file, 'w', encoding='utf-8') as f:
        json.dump(examine_data, f, indent=4)
    logging.info(f"Successfully wrote {len(examine_data)} entries to {new_examine_file}.")
except Exception as e:
    logging.error(f"Failed to write to {new_examine_file}: {e}")
    raise e

logging.info("Process completed successfully.")
