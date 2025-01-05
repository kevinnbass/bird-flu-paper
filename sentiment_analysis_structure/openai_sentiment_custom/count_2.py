import ijson
from collections import Counter, defaultdict
import os
import sys

# Define the categories as before
CATEGORIES = {
    'Scientific News Outlets': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
    'Left News Outlets': ['TheAtlantic', 'The Daily Beast', 'The Intercept', 'Mother Jones', 'MSNBC', 'Slate', 'Vox'],
    'Lean Left News Outlets': ['AP', 'Axios', 'CNN', 'Guardian', 'Business Insider', 'NBCNews', 'NPR', 'NYTimes', 'Politico', 'ProPublica', 'WaPo', 'USA Today'],
    'Centrist News Outlets': ['Reuters', 'MarketWatch', 'Financial Times', 'Newsweek', 'Forbes'],
    'Lean Right News Outlets': ['TheDispatch', 'EpochTimes', 'FoxBusiness', 'WSJ', 'National Review', 'WashTimes'],
    'Right News Outlets': ['Breitbart', 'TheBlaze', 'Daily Mail', 'DailyWire', 'FoxNews', 'NYPost', 'Newsmax']
}

def categorize_media_outlet(outlet):
    for category, outlets in CATEGORIES.items():
        if outlet in outlets:
            return category
    return 'Unknown'

def count_media_outlets_categorized_streaming(json_file_path):
    if not os.path.isfile(json_file_path):
        print(f"Error: File '{json_file_path}' does not exist.")
        sys.exit(1)

    category_counter = defaultdict(Counter)
    total_entries = 0

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            objects = ijson.items(file, 'item')
            for entry in objects:
                total_entries += 1
                media_outlet = entry.get('media_outlet', 'Unknown')
                category = categorize_media_outlet(media_outlet)
                category_counter[category][media_outlet] += 1
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        sys.exit(1)

    # Display the counts as before
    print("Counts of entries per 'media_outlet' categorized:\n")
    for category in CATEGORIES.keys():
        print(f"{category}:")
        outlets = CATEGORIES[category]
        for outlet in outlets:
            count = category_counter[category][outlet]
            print(f"  - {outlet}: {count}")
        print()

    # Handle 'Unknown' category if there are any
    if category_counter['Unknown']:
        print("Unknown Media Outlets:")
        for outlet, count in category_counter['Unknown'].items():
            print(f"  - {outlet}: {count}")
        print()

    print(f"Total number of entries: {total_entries}")

if __name__ == "__main__":
    json_file = 'all_articles_bird_flu_project_for_sa.json'
    count_media_outlets_categorized_streaming(json_file)
