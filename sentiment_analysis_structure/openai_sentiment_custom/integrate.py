import json

sentiment_analysis_file = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'
examine_complete_file = 'processed_all_articles_examine_complete.json'
new_examine_file = 'processed_all_articles_examine_complete_with_huffpost.json'

# Extract huffpost articles from the sentiment analysis file
huffpost_entries = []
with open(sentiment_analysis_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        article = json.loads(line)
        if article.get("media_outlet") == "huffpost":
            huffpost_entries.append(article)

# Load the existing examine_complete file (we assume it's a JSON array)
with open(examine_complete_file, 'r', encoding='utf-8') as f:
    examine_data = json.load(f)

if not isinstance(examine_data, list):
    raise ValueError(f"{examine_complete_file} should contain a JSON array at the top level.")

# Append huffpost entries to the examine_complete data
examine_data.extend(huffpost_entries)

# Write back the updated data to a new file
with open(new_examine_file, 'w', encoding='utf-8') as f:
    json.dump(examine_data, f, indent=4)
