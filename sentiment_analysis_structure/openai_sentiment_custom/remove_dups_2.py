import json
from collections import OrderedDict
import logging
import sys

# Configure logging to output to terminal (stdout)
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Set these variables before running the script for each file.
IS_JSONL = False
INPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost.json'
OUTPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost_cleaned.json'
REVIEW_FILE = 'similar_title_media_outlet_review_examine.json'

logging.info("Starting duplicate removal and review process.")
logging.info(f"Input file: {INPUT_FILE}")
logging.info(f"Output file: {OUTPUT_FILE}")
logging.info(f"Review file: {REVIEW_FILE}")
logging.info(f"File type: {'JSON lines' if IS_JSONL else 'JSON array'}")

# Step 1: Read input file
articles = []
try:
    if IS_JSONL:
        logging.info("Reading as JSON lines file.")
        count_lines = 0
        with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                article = json.loads(line)
                articles.append(article)
                count_lines += 1
        logging.info(f"Read {count_lines} articles from JSON lines file.")
    else:
        logging.info("Reading as JSON array file.")
        with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
            articles = json.load(fin)
        if not isinstance(articles, list):
            raise ValueError("Expected a JSON array at the top level.")
        logging.info(f"Loaded {len(articles)} articles from JSON array file.")
except Exception as e:
    logging.error(f"Failed to read or parse input file: {e}")
    raise e

# Step 2: Group articles by (title, media_outlet)
logging.info("Grouping articles by (title, media_outlet).")
grouped = OrderedDict()
for art in articles:
    key = (art.get("title"), art.get("media_outlet"))
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(art)

logging.info(f"Grouped into {len(grouped)} (title, media_outlet) combinations.")

# Step 3: Remove exact duplicates and identify review groups
logging.info("Removing exact duplicates and identifying review groups.")
review_list = []
cleaned_articles = []
total_exact_duplicates_removed = 0
total_review_groups = 0
total_review_articles = 0

for (title, media_outlet), group in grouped.items():
    seen_fulltexts = {}
    unique_articles = []
    exact_duplicates_for_this_group = 0
    
    for art in group:
        ft = art.get("fulltext")
        if ft not in seen_fulltexts:
            seen_fulltexts[ft] = True
            unique_articles.append(art)
        else:
            exact_duplicates_for_this_group += 1
    
    total_exact_duplicates_removed += exact_duplicates_for_this_group
    
    # Check if multiple unique fulltexts exist for this key
    if len(seen_fulltexts) > 1:
        # multiple different fulltexts under the same (title, media_outlet)
        # add them to review file
        review_list.extend(unique_articles)
        total_review_groups += 1
        total_review_articles += len(unique_articles)
    
    cleaned_articles.extend(unique_articles)

logging.info(f"Exact duplicates removed: {total_exact_duplicates_removed}")
logging.info(f"Number of review groups (same title/media_outlet, multiple fulltexts): {total_review_groups}")
logging.info(f"Number of articles in review: {total_review_articles}")

# Step 4: Write the cleaned output
logging.info(f"Writing cleaned articles to {OUTPUT_FILE}.")
try:
    if IS_JSONL:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            for art in cleaned_articles:
                fout.write(json.dumps(art) + "\n")
    else:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            json.dump(cleaned_articles, fout, indent=4)
    logging.info(f"Wrote {len(cleaned_articles)} cleaned articles to {OUTPUT_FILE}.")
except Exception as e:
    logging.error(f"Failed to write cleaned output: {e}")
    raise e

# Step 5: Write the review file
if review_list:
    logging.info(f"Writing {len(review_list)} articles needing review to {REVIEW_FILE}.")
    try:
        with open(REVIEW_FILE, 'w', encoding='utf-8') as freview:
            json.dump(review_list, freview, indent=4)
        logging.info("Review file written successfully.")
    except Exception as e:
        logging.error(f"Failed to write review file: {e}")
        raise e
else:
    logging.info("No articles require review. Creating empty review file.")
    try:
        with open(REVIEW_FILE, 'w', encoding='utf-8') as freview:
            json.dump([], freview, indent=4)
        logging.info("Empty review file created.")
    except Exception as e:
        logging.error(f"Failed to write empty review file: {e}")
        raise e

logging.info("Process completed successfully.")
