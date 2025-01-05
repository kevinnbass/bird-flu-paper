import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Input JSON file produced by the previous script
INPUT_JSON_FILE = 'final_processed_all_articles.json'

# Output directory for the graphs
OUTPUT_DIR = 'graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the categories and their respective outlets
CATEGORIES = {
    'Scientific News Outlets': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
    'Left News Outlets': ['TheAtlantic', 'The Daily Beast', 'The Intercept', 'Mother Jones', 'MSNBC', 'Slate', 'Vox'],
    'Lean Left News Outlets': ['AP', 'Axios', 'CNN', 'Guardian', 'Business Insider', 'NBCNews', 'NPR', 'NYTimes', 'Politico', 'ProPublica', 'WaPo', 'USA Today'],
    'Centrist News Outlets': ['Reuters', 'MarketWatch', 'Financial Times', 'Newsweek', 'Forbes'],
    'Lean Right News Outlets': ['TheDispatch', 'EpochTimes', 'FoxBusiness', 'WSJ', 'National Review', 'WashTimes'],
    'Right News Outlets': ['Breitbart', 'TheBlaze', 'Daily Mail', 'DailyWire', 'FoxNews', 'NYPost', 'Newsmax'],
}

# ------------------------------ #
#         Helper Functions       #
# ------------------------------ #

def load_data(json_file):
    """
    Loads the JSON data from the specified file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: List of articles.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_quotations(articles):
    """
    Extracts quotations along with their sentiments and fears from the articles.

    Args:
        articles (list): List of articles.

    Returns:
        pd.DataFrame: DataFrame containing quotations, sentiments, fears, and sources.
    """
    records = []
    for article in articles:
        source = article.get('source', 'Unknown')
        # Identify all quotation and assessment pairs
        quotation_keys = [key for key in article.keys() if key.startswith('quotation_')]
        for q_key in quotation_keys:
            index = q_key.split('_')[1]
            quotation = article.get(q_key, '').strip()
            sentiment = article.get(f'sentiment_{index}', 'neutral').strip().lower()
            fear = article.get(f'fear_{index}', 'neutral').strip().lower()
            
            # Standardize categories
            sentiment = sentiment.replace('!', '').replace('positiv', 'positive').replace('posiive', 'positive')
            fear = fear.replace('fearmonger', 'fearmongering')

            if quotation:  # Only include non-empty quotations
                records.append({
                    'source': source,
                    'quotation': quotation,
                    'sentiment': sentiment,
                    'fear': fear
                })
    df = pd.DataFrame(records)
    return df

def map_outlets_to_categories(df, categories_dict):
    """
    Maps each outlet to its respective category.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        categories_dict (dict): Dictionary mapping categories to outlets.

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'category' column.
    """
    outlet_to_category = {}
    for category, outlets in categories_dict.items():
        for outlet in outlets:
            outlet_to_category[outlet] = category
    # Assign category based on source; if not found, assign 'Other'
    df['category'] = df['source'].map(outlet_to_category).fillna('Other')
    return df

def calculate_proportions(df, group_by, column):
    """
    Calculates the proportion of each category within a specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_by (str): Column name to group by (e.g., 'source' or 'category').
        column (str): Column name whose proportions are to be calculated (e.g., 'sentiment').

    Returns:
        pd.DataFrame: DataFrame with proportions.
    """
    total = df.groupby(group_by).size().reset_index(name='total')
    counts = df.groupby([group_by, column]).size().reset_index(name='count')
    merged = pd.merge(counts, total, on=group_by)
    merged['proportion'] = merged['count'] / merged['total']
    return merged

def plot_cluster_bar(data, group, category, value, title, ylabel, filename, palette=None, order=None):
    """
    Plots a cluster bar graph.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        group (str): The grouping variable (e.g., 'source' or 'category').
        category (str): The categorical variable for clustering (e.g., 'sentiment' or 'fear').
        value (str): The numerical variable for bar heights (e.g., 'proportion').
        title (str): The title of the plot.
        ylabel (str): The label for the Y-axis.
        filename (str): The filename to save the plot.
        palette (dict or list, optional): Color palette for the bars.
        order (list, optional): Order of categories.
    """
    plt.figure(figsize=(15, 8))
    sns.barplot(data=data, x=group, y=value, hue=category, palette=palette, order=order)
    plt.title(title, fontsize=16)
    plt.xlabel(group.capitalize(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=category.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# ------------------------------ #
#             Main               #
# ------------------------------ #

def main():
    # Configure logging
    logging.basicConfig(
        filename='sentiment_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load data
    print("Loading data...")
    articles = load_data(INPUT_JSON_FILE)
    print(f"Total articles loaded: {len(articles)}\n")

    # Extract quotations
    print("Extracting quotations...")
    df = extract_quotations(articles)
    print(f"Total quotations extracted: {len(df)}\n")

    if df.empty:
        print("No quotations found in the data. Exiting.")
        return

    # Map outlets to categories
    df = map_outlets_to_categories(df, CATEGORIES)
    print("Mapped outlets to categories.\n")

    # Inspect unique sentiment and fear values
    unique_sentiments = df['sentiment'].unique()
    unique_fears = df['fear'].unique()
    print("Unique Sentiment Values:", unique_sentiments)
    print("Unique Fear Values:", unique_fears)

    # Check for unexpected sentiment values
    expected_sentiments = set(['positive', 'neutral', 'negative'])
    unexpected_sentiments = set(unique_sentiments) - expected_sentiments
    if unexpected_sentiments:
        print(f"Warning: Unexpected sentiment categories found: {unexpected_sentiments}")
        logging.warning(f"Unexpected sentiment categories found: {unexpected_sentiments}")

    # Check for unexpected fear values
    expected_fears = set(['reassuring', 'neutral', 'fearmongering'])
    unexpected_fears = set(unique_fears) - expected_fears
    if unexpected_fears:
        print(f"Warning: Unexpected fear categories found: {unexpected_fears}")
        logging.warning(f"Unexpected fear categories found: {unexpected_fears}")

    print()

    # -------------------------- #
    #      Sentiment Analysis     #
    # -------------------------- #

    print("Calculating sentiment proportions by outlet...")
    sentiment_outlet = calculate_proportions(df, group_by='source', column='sentiment')
    print("Sentiment proportions by outlet calculated.\n")

    print("Calculating sentiment proportions by category...")
    sentiment_category = calculate_proportions(df, group_by='category', column='sentiment')
    print("Sentiment proportions by category calculated.\n")

    # -------------------------- #
    #         Fear Analysis      #
    # -------------------------- #

    print("Calculating fear proportions by outlet...")
    fear_outlet = calculate_proportions(df, group_by='source', column='fear')
    print("Fear proportions by outlet calculated.\n")

    print("Calculating fear proportions by category...")
    fear_category = calculate_proportions(df, group_by='category', column='fear')
    print("Fear proportions by category calculated.\n")

    # -------------------------- #
    #           Plotting          #
    # -------------------------- #

    # Define order for sentiments and fears for consistent coloring
    sentiment_order = ['positive', 'neutral', 'negative']
    fear_order = ['reassuring', 'neutral', 'fearmongering']

    # Define color palettes
    sentiment_palette = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    fear_palette = {'reassuring': 'blue', 'neutral': 'gray', 'fearmongering': 'orange'}

    # Plot Sentiment by Outlet
    print("Plotting sentiment proportions by outlet...")
    plot_cluster_bar(
        data=sentiment_outlet,
        group='source',
        category='sentiment',
        value='proportion',
        title='Proportion of Sentiments by News Outlet',
        ylabel='Proportion',
        filename='sentiment_by_outlet.png',
        palette=sentiment_palette,
        order=sentiment_order
    )
    print("Sentiment proportions by outlet plotted and saved.\n")

    # Plot Sentiment by Category
    print("Plotting sentiment proportions by category...")
    plot_cluster_bar(
        data=sentiment_category,
        group='category',
        category='sentiment',
        value='proportion',
        title='Proportion of Sentiments by News Category',
        ylabel='Proportion',
        filename='sentiment_by_category.png',
        palette=sentiment_palette,
        order=sentiment_order
    )
    print("Sentiment proportions by category plotted and saved.\n")

    # Plot Fear by Outlet
    print("Plotting fear proportions by outlet...")
    plot_cluster_bar(
        data=fear_outlet,
        group='source',
        category='fear',
        value='proportion',
        title='Proportion of Fear Categories by News Outlet',
        ylabel='Proportion',
        filename='fear_by_outlet.png',
        palette=fear_palette,
        order=fear_order
    )
    print("Fear proportions by outlet plotted and saved.\n")

    # Plot Fear by Category
    print("Plotting fear proportions by category...")
    plot_cluster_bar(
        data=fear_category,
        group='category',
        category='fear',
        value='proportion',
        title='Proportion of Fear Categories by News Category',
        ylabel='Proportion',
        filename='fear_by_category.png',
        palette=fear_palette,
        order=fear_order
    )
    print("Fear proportions by category plotted and saved.\n")

    print(f"All graphs have been saved in the '{OUTPUT_DIR}' directory.")
    logging.info("All plots generated successfully.")

if __name__ == "__main__":
    main()
