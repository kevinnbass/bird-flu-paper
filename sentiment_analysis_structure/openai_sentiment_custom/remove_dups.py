import json
from collections import OrderedDict

# Adjust these variables depending on which file you run it on
# For processed_all_articles_examine_complete_with_huffpost.json (JSON array):
# IS_JSONL = False
# INPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost.json'
# OUTPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost_cleaned.json'
# REVIEW_FILE = 'similar_title_media_outlet_review_examine.json'

# For processed_all_articles_with_fulltext_sentiment_analysis.jsonl (JSON lines):
# IS_JSONL = True
# INPUT_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'
# OUTPUT_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis_cleaned.jsonl'
# REVIEW_FILE = 'similar_title_media_outlet_review_sentiment.json'

IS_JSONL = False
INPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost.json'
OUTPUT_FILE = 'processed_all_articles_examine_complete_with_huffpost_cleaned.json'
REVIEW_FILE = 'similar_title_media_outlet_review_examine.json'


# Step 1: Read the input file
articles = []
if IS_JSONL:
    # JSON Lines file
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            articles.append(article)
else:
    # JSON array file
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        articles = json.load(fin)
        if not isinstance(articles, list):
            raise ValueError("Expected a JSON array at the top level")

# Step 2: Group articles by (title, media_outlet)
grouped = OrderedDict()
for art in articles:
    key = (art.get("title"), art.get("media_outlet"))
    if key not in grouped:
        grouped[key] = []
    grouped[key].append(art)

# Step 3: Remove exact duplicates and identify groups with multiple unique fulltexts
review_list = []
cleaned_articles = []

for (title, media_outlet), group in grouped.items():
    seen_fulltexts = {}
    unique_articles = []
    for art in group:
        ft = art.get("fulltext")
        if ft not in seen_fulltexts:
            # First time seeing this fulltext under (title, media_outlet)
            seen_fulltexts[ft] = True
            unique_articles.append(art)
        else:
            # This is an exact duplicate, skip it
            pass
    
    # If more than one unique fulltext is found for the same (title, media_outlet),
    # all those articles are added to the review file for manual inspection.
    if len(seen_fulltexts) > 1:
        for a in unique_articles:
            review_list.append(a)
    
    # Add the unique articles to the final list
    cleaned_articles.extend(unique_articles)

# Step 4: Write the cleaned output
if IS_JSONL:
    # JSON lines output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for art in cleaned_articles:
            fout.write(json.dumps(art) + "\n")
else:
    # JSON array output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        json.dump(cleaned_articles, fout, indent=4)

# Step 5: Write the review file (only if we have entries)
if review_list:
    with open(REVIEW_FILE, 'w', encoding='utf-8') as freview:
        json.dump(review_list, freview, indent=4)
else:
    # If no entries need review, you could optionally create an empty file or skip
    with open(REVIEW_FILE, 'w', encoding='utf-8') as freview:
        json.dump([], freview, indent=4)
