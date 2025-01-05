import json

# Fields to carry over from processed_all_articles_fixed_4.jsonl
FIELDS_TO_APPEND = [
    "anger_fulltext_intensity",
    "anticipation_fulltext_intensity",
    "disgust_fulltext_intensity",
    "fear_fulltext_intensity",
    "joy_fulltext_intensity",
    "negative_fulltext_intensity",
    "positive_fulltext_intensity",
    "sadness_fulltext_intensity",
    "surprise_fulltext_intensity",
    "trust_fulltext_intensity",
    "anger_title_intensity",
    "anticipation_title_intensity",
    "disgust_title_intensity",
    "fear_title_intensity",
    "joy_title_intensity",
    "negative_title_intensity",
    "positive_title_intensity",
    "sadness_title_intensity",
    "surprise_title_intensity",
    "trust_title_intensity",
]

def merge_files(
    in_file_3="processed_all_articles_fixed_3.jsonl",
    in_file_4="processed_all_articles_fixed_4.jsonl",
    out_file_5="processed_all_articles_fixed_5.jsonl",
):
    """
    1. Copy all contents from processed_all_articles_fixed_3.jsonl into a list/dict structure.
    2. Read processed_all_articles_fixed_4.jsonl, match entries on (title, media_outlet).
       For matching entries, copy the intensity fields into the entries from #3.
    3. Write merged records to processed_all_articles_fixed_5.jsonl.
    """

    # Step 1: Read processed_all_articles_fixed_3.jsonl
    #         Store each record in a list and also keep a dict for fast lookups.
    data_3_list = []
    data_3_map = {}  # key: (title, media_outlet), value: record
    with open(in_file_3, "r", encoding="utf-8") as f3:
        for line in f3:
            record = json.loads(line)
            # Construct a lookup key
            key = (record.get("title"), record.get("media_outlet"))
            data_3_list.append(record)
            data_3_map[key] = record

    # Step 2: Read processed_all_articles_fixed_4.jsonl and copy the relevant fields.
    with open(in_file_4, "r", encoding="utf-8") as f4:
        for line in f4:
            record = json.loads(line)
            key = (record.get("title"), record.get("media_outlet"))
            if key in data_3_map:
                # For matching entries, copy the specified fields
                for field in FIELDS_TO_APPEND:
                    if field in record:
                        data_3_map[key][field] = record[field]

    # Step 3: Write everything out to processed_all_articles_fixed_5.jsonl
    with open(out_file_5, "w", encoding="utf-8") as f5:
        for rec in data_3_list:
            f5.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    merge_files()
