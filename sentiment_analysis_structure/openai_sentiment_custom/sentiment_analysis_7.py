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

# Desired order of categories
desired_category_order = ['Scientific', 'Left', 'Lean Left', 'Center', 'Lean Right', 'Right']

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
        source = article.get('media_outlet', 'Unknown').strip()
        # Identify all quotation and assessment pairs
        quotation_keys = [key for key in article.keys() if key.startswith('quotation_')]
        for q_key in quotation_keys:
            index = q_key.split('_')[1]
            quotation = article.get(q_key, '').strip()
            sentiment = article.get(f'sentiment_{index}', 'neutral').strip().lower()
            fear = article.get(f'fear_{index}', 'neutral').strip().lower()
            
            # ----------------------------
            # Data Cleaning for Sentiment
            # ----------------------------
            # Correct typographical errors and misassigned categories in sentiment
            sentiment_corrections = {
                'positivee': 'positive',
                'posiive': 'positive',
                'positiv': 'positive',
                'reassuring': 'neutral',  # 'reassuring' should not be in sentiment
                # Add more corrections as needed
            }
            sentiment = sentiment_corrections.get(sentiment, sentiment)
            
            # ----------------------------
            # Data Cleaning for Fear
            # ----------------------------
            # Correct typographical errors in fear
            fear_corrections = {
                'fearmongeringing': 'fearmongering',
                'fearmonger': 'fearmongering',
                # Add more corrections as needed
            }
            fear = fear_corrections.get(fear, fear)
            
            # Ensure 'reassuring' is only in fear, not in sentiment
            if sentiment == 'reassuring':
                sentiment = 'neutral'  # or another appropriate value
                logging.warning(f"Corrected 'reassuring' in sentiment to 'neutral' for source: {source}")
            
            if quotation:  # Only include non-empty quotations
                records.append({
                    'source': source,
                    'quotation': quotation,
                    'sentiment': sentiment,
                    'fear': fear
                })
    df = pd.DataFrame(records)
    
    # Print the first few rows for verification
    print("\nSample of Extracted and Cleaned Data:")
    print(df.head(10))
    
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
            outlet_to_category[outlet.lower().strip()] = category  # Convert to lowercase and strip spaces
    
    # Assign category based on source; if not found, assign 'Other'
    df['category'] = df['source'].str.lower().map(outlet_to_category).fillna('Other')
    
    # Print unique sources and their mapped categories
    print("\nMapping of Media Outlets to Categories:")
    mapping = df.groupby(['source', 'category']).size().reset_index(name='count')
    print(mapping.head(10))
    
    return df

def rename_categories(df):
    """
    Renames the categories to simpler labels for plotting.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: Updated DataFrame with renamed 'category' column.
    """
    category_mapping = {
        'Scientific News Outlets': 'Scientific',
        'Left News Outlets': 'Left',
        'Lean Left News Outlets': 'Lean Left',
        'Centrist News Outlets': 'Center',
        'Lean Right News Outlets': 'Lean Right',
        'Right News Outlets': 'Right'
    }
    df['category'] = df['category'].replace(category_mapping)
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
    merged['proportion'] = merged.apply(lambda row: row['count'] / row['total'] if row['total'] > 0 else 0, axis=1)
    
    # Print the first few rows for verification
    print(f"\nProportions of {column.capitalize()} by {group_by.capitalize()}:")
    print(merged.head(10))
    
    return merged

def generate_palette(categories):
    """
    Generates a color palette for the given categories.

    Args:
        categories (list): List of category names.

    Returns:
        dict: Dictionary mapping categories to colors.
    """
    palette = sns.color_palette("hsv", len(categories))
    return dict(zip(categories, palette))

def plot_cluster_bar(data, group, category, value, title, ylabel, filename, palette=None, hue_order=None, order=None):
    """
    Plots a clustered bar graph.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        group (str): The grouping variable (e.g., 'source' or 'category').
        category (str): The categorical variable for clustering (e.g., 'sentiment' or 'fear').
        value (str): The numerical variable for bar heights (e.g., 'proportion').
        title (str): The title of the plot.
        ylabel (str): The label for the Y-axis.
        filename (str): The filename to save the plot.
        palette (dict or list, optional): Color palette for the bars.
        hue_order (list, optional): Order of hue categories.
        order (list, optional): Order of x-axis categories.
    """
    if data.empty:
        print(f"Warning: No data available for plotting '{title}'. Skipping plot.")
        logging.warning(f"No data available for plotting '{title}'.")
        return
    
    # Print data to be plotted
    print(f"\nData to be plotted for '{title}':")
    print(data.head(10))
    
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=data,
        x=group,
        y=value,
        hue=category,
        palette=palette,
        hue_order=hue_order,
        order=order
    )
    plt.title(title, fontsize=16)
    plt.xlabel(group.capitalize(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=category.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Plot saved as {filename}.")

def validate_categories(df):
    """
    Validates that sentiment and fear categories are as expected.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        None
    """
    unique_sentiments = df['sentiment'].unique()
    unique_fears = df['fear'].unique()
    print("\nUnique Sentiment Values:", unique_sentiments)
    print("Unique Fear Values:", unique_fears)
    
    # Define expected categories
    expected_sentiments = {'positive', 'neutral', 'negative'}
    expected_fears = {'reassuring', 'neutral', 'fearmongering'}
    
    # Identify unexpected categories
    unexpected_sentiments = set(unique_sentiments) - expected_sentiments
    unexpected_fears = set(unique_fears) - expected_fears
    
    if unexpected_sentiments:
        print(f"Warning: Unexpected sentiment categories found: {unexpected_sentiments}")
        logging.warning(f"Unexpected sentiment categories found: {unexpected_sentiments}")
    
    if unexpected_fears:
        print(f"Warning: Unexpected fear categories found: {unexpected_fears}")
        logging.warning(f"Unexpected fear categories found: {unexpected_fears}")

def check_remaining_unexpected_values(df):
    """
    Checks for any remaining unexpected values in sentiment and fear columns.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        None
    """
    expected_sentiments = {'positive', 'neutral', 'negative'}
    expected_fears = {'reassuring', 'neutral', 'fearmongering'}
    
    actual_sentiments = set(df['sentiment'].unique())
    actual_fears = set(df['fear'].unique())
    
    unexpected_sentiments = actual_sentiments - expected_sentiments
    unexpected_fears = actual_fears - expected_fears
    
    if unexpected_sentiments:
        print(f"\nUnexpected sentiment categories: {unexpected_sentiments}")
    else:
        print("\nNo unexpected sentiment categories found.")
    
    if unexpected_fears:
        print(f"Unexpected fear categories: {unexpected_fears}")
    else:
        print("No unexpected fear categories found.")

def print_complete_category_sums(df, group_by, column, label):
    """
    Prints the complete sum of categories for a given group and column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_by (str): Column name to group by (e.g., 'source' or 'category').
        column (str): Column name to sum (e.g., 'sentiment').
        label (str): Label for the print statement.

    Returns:
        None
    """
    counts = df.groupby([group_by, column]).size().reset_index(name='count')
    print(f"\n{label} Counts by {group_by.capitalize()}:")
    for _, row in counts.iterrows():
        print(f"{row[group_by]} - {row[column]}: {row['count']}")

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
    try:
        articles = load_data(INPUT_JSON_FILE)
        print(f"Total articles loaded: {len(articles)}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return

    # Extract quotations
    print("Extracting quotations...")
    try:
        df = extract_quotations(articles)
        print(f"Total quotations extracted: {len(df)}\n")
    except Exception as e:
        print(f"Error extracting quotations: {e}")
        logging.error(f"Error extracting quotations: {e}")
        return

    if df.empty:
        print("No quotations found in the data. Exiting.")
        logging.info("No quotations found in the data. Exiting.")
        return

    # Map outlets to categories
    print("Mapping outlets to categories...")
    try:
        df = map_outlets_to_categories(df, CATEGORIES)
        print("Mapped outlets to categories.\n")
    except Exception as e:
        print(f"Error mapping outlets to categories: {e}")
        logging.error(f"Error mapping outlets to categories: {e}")
        return

    # Rename categories for clarity
    try:
        df = rename_categories(df)
        print("Renamed categories for clarity.\n")
    except Exception as e:
        print(f"Error renaming categories: {e}")
        logging.error(f"Error renaming categories: {e}")

    # Validate categories
    try:
        validate_categories(df)
    except Exception as e:
        print(f"Error validating categories: {e}")
        logging.error(f"Error validating categories: {e}")

    # Check for any remaining unexpected values
    try:
        check_remaining_unexpected_values(df)
    except Exception as e:
        print(f"Error checking unexpected values: {e}")
        logging.error(f"Error checking unexpected values: {e}")

    print()

    # Print Complete Counts Before Proportion Calculation
    print_complete_category_sums(df, group_by='source', column='sentiment', label='Sentiment')
    print_complete_category_sums(df, group_by='category', column='sentiment', label='Sentiment by Category')
    print_complete_category_sums(df, group_by='source', column='fear', label='Fear')
    print_complete_category_sums(df, group_by='category', column='fear', label='Fear by Category')

    # -------------------------- #
    #      Sentiment Analysis     #
    # -------------------------- #

    print("\nCalculating sentiment proportions by outlet...")
    try:
        sentiment_outlet = calculate_proportions(df, group_by='source', column='sentiment')
        print("Sentiment proportions by outlet calculated.\n")
    except Exception as e:
        print(f"Error calculating sentiment proportions by outlet: {e}")
        logging.error(f"Error calculating sentiment proportions by outlet: {e}")
        sentiment_outlet = pd.DataFrame()

    print("Calculating sentiment proportions by category...")
    try:
        sentiment_category = calculate_proportions(df, group_by='category', column='sentiment')
        print("Sentiment proportions by category calculated.\n")
    except Exception as e:
        print(f"Error calculating sentiment proportions by category: {e}")
        logging.error(f"Error calculating sentiment proportions by category: {e}")
        sentiment_category = pd.DataFrame()

    # -------------------------- #
    #         Fear Analysis      #
    # -------------------------- #

    print("Calculating fear proportions by outlet...")
    try:
        fear_outlet = calculate_proportions(df, group_by='source', column='fear')
        print("Fear proportions by outlet calculated.\n")
    except Exception as e:
        print(f"Error calculating fear proportions by outlet: {e}")
        logging.error(f"Error calculating fear proportions by outlet: {e}")
        fear_outlet = pd.DataFrame()

    print("Calculating fear proportions by category...")
    try:
        fear_category = calculate_proportions(df, group_by='category', column='fear')
        print("Fear proportions by category calculated.\n")
    except Exception as e:
        print(f"Error calculating fear proportions by category: {e}")
        logging.error(f"Error calculating fear proportions by category: {e}")
        fear_category = pd.DataFrame()

    # -------------------------- #
    #           Plotting          #
    # -------------------------- #

    # Define order for sentiments and fears for consistent coloring
    sentiment_order = ['positive', 'neutral', 'negative']
    fear_order = ['reassuring', 'neutral', 'fearmongering']

    # Define desired order for categories
    desired_order = ['Scientific', 'Left', 'Lean Left', 'Center', 'Lean Right', 'Right']

    # Generate dynamic palettes
    sentiment_palette = generate_palette(sentiment_order)
    fear_palette = generate_palette(fear_order)

    # Plot Sentiment by Outlet
    print("Plotting sentiment proportions by outlet...")
    try:
        plot_cluster_bar(
            data=sentiment_outlet,
            group='source',
            category='sentiment',
            value='proportion',
            title='Proportion of Sentiments by News Outlet',
            ylabel='Proportion',
            filename='sentiment_by_outlet.png',
            palette=sentiment_palette,
            hue_order=sentiment_order  # Hue order for sentiment
            # No specific order for sources
        )
    except Exception as e:
        print(f"Error plotting sentiment proportions by outlet: {e}")
        logging.error(f"Error plotting sentiment proportions by outlet: {e}")

    # Plot Sentiment by Category
    print("Plotting sentiment proportions by category...")
    try:
        plot_cluster_bar(
            data=sentiment_category,
            group='category',
            category='sentiment',
            value='proportion',
            title='Proportion of Sentiments by News Category',
            ylabel='Proportion',
            filename='sentiment_by_category.png',
            palette=sentiment_palette,
            hue_order=sentiment_order,  # Hue order for sentiment
            order=desired_order         # X-axis order for categories
        )
    except Exception as e:
        print(f"Error plotting sentiment proportions by category: {e}")
        logging.error(f"Error plotting sentiment proportions by category: {e}")

    # Plot Fear by Outlet
    print("Plotting fear proportions by outlet...")
    try:
        plot_cluster_bar(
            data=fear_outlet,
            group='source',
            category='fear',
            value='proportion',
            title='Proportion of Fear Categories by News Outlet',
            ylabel='Proportion',
            filename='fear_by_outlet.png',
            palette=fear_palette,
            hue_order=fear_order  # Hue order for fear
            # No specific order for sources
        )
    except Exception as e:
        print(f"Error plotting fear proportions by outlet: {e}")
        logging.error(f"Error plotting fear proportions by outlet: {e}")

    # Plot Fear by Category
    print("Plotting fear proportions by category...")
    try:
        plot_cluster_bar(
            data=fear_category,
            group='category',
            category='fear',
            value='proportion',
            title='Proportion of Fear Categories by News Category',
            ylabel='Proportion',
            filename='fear_by_category.png',
            palette=fear_palette,
            hue_order=fear_order,       # Hue order for fear
            order=desired_order         # X-axis order for categories
        )
    except Exception as e:
        print(f"Error plotting fear proportions by category: {e}")
        logging.error(f"Error plotting fear proportions by category: {e}")

    print(f"\nAll graphs have been saved in the '{OUTPUT_DIR}' directory.")
    logging.info("All plots generated successfully.")

if __name__ == "__main__":
    main()
