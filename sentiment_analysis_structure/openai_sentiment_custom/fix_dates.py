import json
import sys

input_file = 'processed_all_articles_merged.jsonl'
output_file = 'processed_all_articles_fixed.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)
        if "date" in obj:
            # Extract just the date portion (YYYY-MM-DD)
            # Assuming the date is always at least in the form YYYY-MM-DD
            # even if there is additional time info after that.
            date_str = obj["date"]
            # If date_str is shorter than 10 chars or differently formatted, 
            # you might need more robust logic. This assumes consistent formatting.
            obj["date"] = date_str[:10]  
        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
