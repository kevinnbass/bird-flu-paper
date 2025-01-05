import json
import logging
import sys
import pandas as pd
from collections import OrderedDict

# --------------------------------------------------
# 1) Convert readability_indices_raw_data.xlsx -> JSONL
# --------------------------------------------------

def convert_xlsx_to_jsonl(excel_path, jsonl_path):
    """
    Reads an Excel file and writes each row to a JSON Lines file.
    Converts all columns to string to avoid "Timestamp not serializable" errors.
    """
    # Read Excel into a DataFrame
    df = pd.read_excel(excel_path)
    
    # Convert everything to string (quick fix to avoid JSON serialization errors)
    df = df.astype(str)

    # Write out line by line to JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            record = row.to_dict()
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

# --------------------------------------------------
# 2) Deduplicate by “longest fulltext” logic
# --------------------------------------------------

def remove_duplicates_in_jsonl(input_jsonl, output_jsonl):
    """
    Reads a JSON Lines file, groups records by (title, media_outlet),
    and keeps only the record with the longest 'fulltext' within each group.

    Writes the deduplicated records to a new JSON Lines file.
    """
    # Configure logging to output to terminal (stdout)
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    logging.info("------------------------------------------------------------")
    logging.info("Starting longest-fulltext selection process.")
    logging.info(f"Input file: {input_jsonl}")
    logging.info(f"Output file: {output_jsonl}")

    # Step 1: Read input file
    articles = []
    count_lines = 0
    try:
        logging.info("Reading as JSON lines file.")
        with open(input_jsonl, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                article = json.loads(line)
                articles.append(article)
                count_lines += 1
        logging.info(f"Read {count_lines} articles from JSON lines file.")
    except Exception as e:
        logging.error(f"Failed to read or parse input file: {e}")
        raise e

    # Step 2: Group articles by (title, media_outlet)
    logging.info("Grouping articles by (title, media_outlet).")
    grouped = OrderedDict()
    for art in articles:
        title = art.get("title", "")
        media_outlet = art.get("media_outlet", "")
        key = (title, media_outlet)
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

        # Keep the record with the longest fulltext
        cleaned_articles.append(longest_article)

        # Count how many were removed
        total_removed += (len(group) - 1)

    logging.info(f"Removed {total_removed} articles that did not have the longest fulltext.")

    # Step 4: Write the cleaned output
    logging.info(f"Writing cleaned articles to {output_jsonl}.")
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as fout:
            for art in cleaned_articles:
                fout.write(json.dumps(art, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(cleaned_articles)} cleaned articles to {output_jsonl}.")
    except Exception as e:
        logging.error(f"Failed to write cleaned output: {e}")
        raise e

    logging.info("Process completed successfully.")

# --------------------------------------------------
# 3) Run both steps in sequence
# --------------------------------------------------

if __name__ == "__main__":
    excel_file = "readability_indices_raw_data.xlsx"
    intermediate_jsonl = "readability_indices_raw_data.jsonl"
    output_jsonl = "readability_indices_raw_data_cleaned.jsonl"

    # Step 1: Convert Excel -> JSONL (with fix for Timestamps)
    convert_xlsx_to_jsonl(excel_file, intermediate_jsonl)

    # Step 2: Deduplicate by longest fulltext
    remove_duplicates_in_jsonl(intermediate_jsonl, output_jsonl)

    print(f"Done. Cleaned data is in: {output_jsonl}")
