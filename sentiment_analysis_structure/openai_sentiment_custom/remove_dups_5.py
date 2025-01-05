import json
import logging
import sys

# Configure logging to output to terminal (stdout)
logging.basicConfig(
    stream=sys.stdout,          # Send logs to stdout (terminal)
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO          # INFO level shows basic progress information
)

# Filenames
EXAMINE_FILE = 'processed_all_articles_examine_complete_with_huffpost_cleaned.json'
SENTIMENT_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis_cleaned.jsonl'
MERGED_FILE = 'processed_all_articles_merged.jsonl'
UNMATCHED_FILE = 'unmatched_log.json'

logging.info("Starting the merging process.")

# Step 1: Load examine data
logging.info(f"Loading examine_data from {EXAMINE_FILE}.")
try:
    with open(EXAMINE_FILE, 'r', encoding='utf-8') as f:
        examine_data = json.load(f)
    logging.info(f"Loaded {len(examine_data)} entries from examine_data.")
except Exception as e:
    logging.error(f"Failed to load examine_data from {EXAMINE_FILE}: {e}")
    raise e

# Step 2: Create a lookup dictionary from (title, media_outlet) -> (high_rate_2, nuance_2)
logging.info("Creating lookup dictionary from examine_data.")
examine_lookup = {}
duplicates_found = 0
for article in examine_data:
    key = (article.get("title"), article.get("media_outlet"))
    # Store the fields if they exist
    # If duplicate keys occur, the last one encountered will overwrite the previous
    if key in examine_lookup:
        duplicates_found += 1
    examine_lookup[key] = {
        "high_rate_2": article.get("high_rate_2", None),
        "nuance_2": article.get("nuance_2", None)
    }

logging.info(f"Created lookup with {len(examine_lookup)} unique keys.")
if duplicates_found > 0:
    logging.info(f"Warning: Overwritten {duplicates_found} duplicate keys in examine_lookup.")

updated_articles = []
unmatched_log = []

logging.info(f"Opening {SENTIMENT_FILE} to merge data.")
count_matches = 0
count_total = 0

# Step 3: Read sentiment file and update with examine data
try:
    with open(SENTIMENT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            count_total += 1

            # Match on title and media_outlet
            key = (article.get("title"), article.get("media_outlet"))
            if key in examine_lookup:
                # Update the article fields
                article["high_rate_2"] = examine_lookup[key]["high_rate_2"]
                article["nuance_2"] = examine_lookup[key]["nuance_2"]
                # Remove the matched entry from examine_lookup so we know which ones are left unmatched
                del examine_lookup[key]
                count_matches += 1

            updated_articles.append(article)

    logging.info(f"Processed {count_total} articles from sentiment analysis file.")
    logging.info(f"Matched and updated {count_matches} articles.")
except Exception as e:
    logging.error(f"Error reading {SENTIMENT_FILE}: {e}")
    raise e

# Step 4: Write the merged articles to a new file
logging.info(f"Writing updated articles to {MERGED_FILE}.")
try:
    with open(MERGED_FILE, 'w', encoding='utf-8') as fout:
        for art in updated_articles:
            fout.write(json.dumps(art) + "\n")
    logging.info(f"Wrote {len(updated_articles)} articles to {MERGED_FILE}.")
except Exception as e:
    logging.error(f"Failed to write to {MERGED_FILE}: {e}")
    raise e

# Step 5: Any remaining keys in examine_lookup did not find a match
logging.info("Checking for unmatched entries.")
for key, values in examine_lookup.items():
    unmatched_log.append({
        "title": key[0],
        "media_outlet": key[1],
        "reason": "No matching entry found in the sentiment analysis file"
    })

try:
    with open(UNMATCHED_FILE, 'w', encoding='utf-8') as flog:
        json.dump(unmatched_log, flog, indent=4)
    logging.info(f"Wrote {len(unmatched_log)} unmatched entries to {UNMATCHED_FILE}.")
except Exception as e:
    logging.error(f"Failed to write unmatched log to {UNMATCHED_FILE}: {e}")
    raise e

logging.info("Merging process completed successfully.")
