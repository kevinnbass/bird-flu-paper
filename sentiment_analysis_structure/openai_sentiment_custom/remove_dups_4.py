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

def run_for_file(IS_JSONL, INPUT_FILE, OUTPUT_FILE):
    """
    Process the given file by grouping articles by (title, media_outlet).
    If multiple articles share the same (title, media_outlet) but have different fulltexts,
    keep only the one with the longest fulltext. Remove all other variants.

    Parameters:
        IS_JSONL (bool): True if the input is in JSON Lines format, False if JSON array.
        INPUT_FILE (str): Path to the input file.
        OUTPUT_FILE (str): Path to write the cleaned output.
    """
    logging.info("------------------------------------------------------------")
    logging.info("Starting longest-fulltext selection process for a new file.")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")
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

    # Step 3: For each group, keep only the article with the longest fulltext
    cleaned_articles = []
    total_removed = 0
    for (title, media_outlet), group in grouped.items():
        if len(group) == 1:
            # Only one article, keep it as is
            cleaned_articles.append(group[0])
            continue
        
        # Multiple articles for this (title, media_outlet)
        # Find the longest fulltext
        longest_article = None
        longest_length = -1
        
        for art in group:
            ft = art.get("fulltext", "")
            ft_length = len(ft)
            if ft_length > longest_length:
                longest_length = ft_length
                longest_article = art
        
        # longest_article is now the one with the longest fulltext
        # Add it to the cleaned_articles
        cleaned_articles.append(longest_article)
        
        # Count how many were removed
        removed_count = len(group) - 1
        total_removed += removed_count

    logging.info(f"Removed {total_removed} articles that did not have the longest fulltext.")

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

    logging.info("Process for this file completed successfully.")

#########################
# Main processing logic #
#########################

# Run for the JSON array file (processed_all_articles_examine_complete_with_huffpost.json)
run_for_file(
    IS_JSONL=False,
    INPUT_FILE='processed_all_articles_examine_complete_with_huffpost.json',
    OUTPUT_FILE='processed_all_articles_examine_complete_with_huffpost_cleaned.json'
)

# Run for the JSON lines file (processed_all_articles_with_fulltext_sentiment_analysis.jsonl)
run_for_file(
    IS_JSONL=True,
    INPUT_FILE='processed_all_articles_with_fulltext_sentiment_analysis.jsonl',
    OUTPUT_FILE='processed_all_articles_with_fulltext_sentiment_analysis_cleaned.jsonl'
)

logging.info("All processes completed successfully.")
