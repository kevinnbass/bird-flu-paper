import json
import pandas as pd
import os
from collections import defaultdict

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Input JSON file produced by the previous script
INPUT_JSON_FILE = 'final_processed_all_articles.json'

# Output JSON file
OUTPUT_JSON_FILE = 'sentiment_openai.json'

# Define the categories and their respective outlets
CATEGORIES = {
    'Scientific News Outlets': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
    'Left News Outlets': ['TheAtlantic', 'The Daily Beast', 'HuffPost', 'The Intercept', 'Mother Jones', 'MSNBC', 'Slate', 'Vox'],
    'Lean Left News Outlets': ['AP', 'Axios', 'CNN', 'Guardian', 'Business Insider', 'NBCNews', 'NPR', 'NYTimes', 'Politico', 'ProPublica', 'WaPo', 'USA Today'],
    'Centrist News Outlets': ['Reuters', 'MarketWatch', 'Financial Times', 'Newsweek', 'Forbes'],
    'Lean Right News Outlets': ['TheDispatch', 'EpochTimes', 'FoxBusiness', 'WSJ', 'National Review', 'WashTimes'],
    'Right News Outlets': ['Breitbart', 'TheBlaze', 'Daily Mail', 'DailyWire', 'FoxNews', 'NYPost', 'Newsmax'],
}

# Desired order of categories
DESIRED_CATEGORY_ORDER = [
    'Scientific News Outlets',
    'Left News Outlets',
    'Lean Left News Outlets',
    'Centrist News Outlets',
    'Lean Right News Outlets',
    'Right News Outlets'
]

# ------------------------------ #
#         Helper Functions       #
# ------------------------------ #

def load_data(json_file):
    """
    Loads the JSON data from the specified file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: List of articles (dictionaries).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def map_outlet_to_category(outlet, category_dict):
    """
    Maps a media outlet to its corresponding category.

    Args:
        outlet (str): Name of the media outlet.
        category_dict (dict): Dictionary mapping categories to outlets.

    Returns:
        str or None: Category name if found, else None.
    """
    outlet_lower = outlet.lower().strip()
    for category, outlets in category_dict.items():
        if outlet_lower in [o.lower().strip() for o in outlets]:
            return category
    return None

def initialize_category_counts(categories):
    """
    Initializes the counting structure for each category.

    Args:
        categories (list): List of category names.

    Returns:
        dict: Dictionary with categories as keys and sentiment/fear counters as nested dictionaries.
    """
    counts = {}
    for category in categories:
        counts[category] = {
            'Category': category,
            'Fear_Fearmonger': 0,
            'Fear_Neutral': 0,
            'Fear_Reassuring': 0,
            'Sentiment_Negative': 0,
            'Sentiment_Neutral': 0,
            'Sentiment_Positive': 0,
            # New proportion fields initialized to 0
            'Negative_Proportion': 0.0,
            'Positive_Proportion': 0.0,
            'Fearmonger_Proportion': 0.0,
            'Reassuring_Proportion': 0.0
        }
    return counts

def process_article(article, category_counts, category_dict):
    """
    Processes a single article, updating the category counts.

    Args:
        article (dict): The article data.
        category_counts (dict): The counts dictionary to update.
        category_dict (dict): The categories and their outlets.
    """
    outlet = article.get('media_outlet', 'Unknown').strip()
    category = map_outlet_to_category(outlet, category_dict)
    if not category:
        # Outlet not found in any category; skip
        return

    # Iterate through all keys to find 'fear_x' and 'sentiment_x'
    for key, value in article.items():
        if key.startswith('fear_'):
            fear_value = value.strip().lower()
            if fear_value == 'fearmongering':
                category_counts[category]['Fear_Fearmonger'] += 1
            elif fear_value == 'neutral':
                category_counts[category]['Fear_Neutral'] += 1
            elif fear_value == 'reassuring':
                category_counts[category]['Fear_Reassuring'] += 1
            # You can add more conditions if there are other fear types
        elif key.startswith('sentiment_'):
            sentiment_value = value.strip().lower()
            if sentiment_value == 'negative':
                category_counts[category]['Sentiment_Negative'] += 1
            elif sentiment_value == 'neutral':
                category_counts[category]['Sentiment_Neutral'] += 1
            elif sentiment_value == 'positive':
                category_counts[category]['Sentiment_Positive'] += 1
            # You can add more conditions if there are other sentiment types

def calculate_proportions(category_counts):
    """
    Calculates the required proportions for each category and updates the counts dictionary.

    Args:
        category_counts (dict): The counts dictionary to update.
    """
    for category, counts in category_counts.items():
        sentiment_total = counts['Sentiment_Negative'] + counts['Sentiment_Positive']
        fear_total = counts['Fear_Fearmonger'] + counts['Fear_Reassuring']

        # Calculate Negative Proportion
        if sentiment_total > 0:
            counts['Negative_Proportion'] = counts['Sentiment_Negative'] / sentiment_total
            counts['Positive_Proportion'] = counts['Sentiment_Positive'] / sentiment_total
        else:
            counts['Negative_Proportion'] = 0.0
            counts['Positive_Proportion'] = 0.0

        # Calculate Fearmonger and Reassuring Proportions
        if fear_total > 0:
            counts['Fearmonger_Proportion'] = counts['Fear_Fearmonger'] / fear_total
            counts['Reassuring_Proportion'] = counts['Fear_Reassuring'] / fear_total
        else:
            counts['Fearmonger_Proportion'] = 0.0
            counts['Reassuring_Proportion'] = 0.0

def save_to_json(counts, output_file):
    """
    Saves the counts dictionary to a JSON file.

    Args:
        counts (dict): The counts dictionary.
        output_file (str): The path to the output JSON file.
    """
    # Convert the counts dictionary to a list for JSON serialization
    counts_list = list(counts.values())
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(counts_list, f, indent=4)
    print(f"JSON data saved to {output_file}")

# ------------------------------ #
#             Main               #
# ------------------------------ #

def main():
    # Check if input file exists
    if not os.path.isfile(INPUT_JSON_FILE):
        print(f"Error: The input file '{INPUT_JSON_FILE}' does not exist.")
        return

    # Load data
    print("Loading data...")
    try:
        articles = load_data(INPUT_JSON_FILE)
        print(f"Total articles loaded: {len(articles)}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize category counts
    category_counts = initialize_category_counts(DESIRED_CATEGORY_ORDER)

    # Process each article
    print("Processing articles...")
    for article in articles:
        process_article(article, category_counts, CATEGORIES)
    print("Articles processed.\n")

    # Calculate proportions
    print("Calculating proportions...")
    calculate_proportions(category_counts)
    print("Proportions calculated.\n")

    # Save the counts to JSON
    print(f"Saving counts to '{OUTPUT_JSON_FILE}'...")
    try:
        save_to_json(category_counts, OUTPUT_JSON_FILE)
        print("JSON file created successfully.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

if __name__ == "__main__":
    main()
