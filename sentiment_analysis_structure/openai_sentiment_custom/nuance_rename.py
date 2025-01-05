import json

filename = 'processed_all_articles_examine_complete_with_huffpost.json'

# Load the JSON array from the file
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Iterate through each article and modify fields if media_outlet is HuffPost
for article in data:
    if article.get("media_outlet") == "HuffPost":
        # If "high_rate" exists, rename it to "high_rate_2"
        if "high_rate" in article:
            article["high_rate_2"] = article["high_rate"]
            del article["high_rate"]
        # If "nuance" exists, rename it to "nuance_2"
        if "nuance" in article:
            article["nuance_2"] = article["nuance"]
            del article["nuance"]

# Write the updated data back to the file
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
