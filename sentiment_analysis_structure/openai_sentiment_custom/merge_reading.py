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
EXAMINE_FILE = 'readability_indices_raw_data_cleaned.jsonl'  # Changed
SENTIMENT_FILE = 'processed_all_articles_fixed_2.jsonl'      # Changed
MERGED_FILE = 'processed_all_articles_fixed_3.jsonl'         # Changed
UNMATCHED_FILE = 'unmatched_log.json'

logging.info("Starting the merging process.")

##########################
# Step 1: Load examine data
##########################
logging.info(f"Loading examine_data from {EXAMINE_FILE}.")
try:
    examine_data = []
    with open(EXAMINE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examine_data.append(json.loads(line))
    logging.info(f"Loaded {len(examine_data)} entries from examine_data.")
except Exception as e:
    logging.error(f"Failed to load examine_data from {EXAMINE_FILE}: {e}")
    raise e

#############################################################################
# Step 2: Create a lookup dict: (title, media_outlet) -> { readability fields }
#############################################################################
logging.info("Creating lookup dictionary from examine_data.")
examine_lookup = {}
duplicates_found = 0

for article in examine_data:
    key = (article.get("title"), article.get("media_outlet"))
    # If duplicate keys occur, the last one encountered overwrites the previous
    if key in examine_lookup:
        duplicates_found += 1
    
    # We store only the readability fields we need
    examine_lookup[key] = {
        "flesch_kincaid_grade_global": article.get("flesch_kincaid_grade_global"),
        "gunning_fog_global": article.get("gunning_fog_global")
    }

logging.info(f"Created lookup with {len(examine_lookup)} unique keys.")
if duplicates_found > 0:
    logging.info(f"Warning: Overwritten {duplicates_found} duplicate keys in examine_lookup.")

updated_articles = []
unmatched_log = []

###################################
# Step 3: Merge with Sentiment Data
###################################
logging.info(f"Opening {SENTIMENT_FILE} to merge data.")
count_matches = 0
count_total = 0

try:
    with open(SENTIMENT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            count_total += 1

            # Match on (title, media_outlet)
            key = (article.get("title"), article.get("media_outlet"))
            if key in examine_lookup:
                # Update the article with the two readability fields
                article["flesch_kincaid_grade_global"] = examine_lookup[key]["flesch_kincaid_grade_global"]
                article["gunning_fog_global"] = examine_lookup[key]["gunning_fog_global"]
                
                # Remove matched entry so we can identify truly unmatched later
                del examine_lookup[key]
                count_matches += 1

            updated_articles.append(article)

    logging.info(f"Processed {count_total} articles from sentiment file.")
    logging.info(f"Matched and updated {count_matches} articles.")
except Exception as e:
    logging.error(f"Error reading {SENTIMENT_FILE}: {e}")
    raise e

#####################################################
# Step 4: Write merged articles to the new JSONL file
#####################################################
logging.info(f"Writing updated articles to {MERGED_FILE}.")
try:
    with open(MERGED_FILE, 'w', encoding='utf-8') as fout:
        for art in updated_articles:
            fout.write(json.dumps(art, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(updated_articles)} articles to {MERGED_FILE}.")
except Exception as e:
    logging.error(f"Failed to write to {MERGED_FILE}: {e}")
    raise e

########################################################
# Step 5: Any leftover keys in examine_lookup are unmatched
########################################################
logging.info("Checking for unmatched entries.")
for key, values in examine_lookup.items():
    unmatched_log.append({
        "title": key[0],
        "media_outlet": key[1],
        "reason": "No matching entry found in the sentiment analysis file"
    })

# Save unmatched info as a separate JSON
try:
    with open(UNMATCHED_FILE, 'w', encoding='utf-8') as flog:
        json.dump(unmatched_log, flog, indent=4)
    logging.info(f"Wrote {len(unmatched_log)} unmatched entries to {UNMATCHED_FILE}.")
except Exception as e:
    logging.error(f"Failed to write unmatched log to {UNMATCHED_FILE}: {e}")
    raise e

logging.info("Merging process completed successfully.")
