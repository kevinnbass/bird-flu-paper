import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
from openpyxl.drawing.image import Image as ExcelImage
import re

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Input JSONL file produced by the first script
INPUT_JSONL_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'

# Output directories and files
OUTPUT_DIR = 'graphs_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUTPUT_DIR = 'csv_raw_scores'
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

OUTPUT_EXCEL_FILE = 'analysis_results.xlsx'

# Define the ten sentiment/emotion categories
CATEGORIES = [
    'joy', 'sadness', 'anger', 'fear',
    'surprise', 'disgust', 'trust', 'anticipation',
    'negative_sentiment', 'positive_sentiment'
]

# Define the six media categories and their corresponding outlets
MEDIA_CATEGORIES = {
    'Scientific': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
    'Left': ['TheAtlantic', 'The Daily Beast', 'The Intercept', 'Mother Jones', 'MSNBC', 'Slate', 'Vox', 'HuffPost'],
    'Lean Left': ['AP', 'Axios', 'CNN', 'Guardian', 'Business Insider', 'NBCNews', 'NPR', 'NYTimes', 'Politico', 'ProPublica', 'WaPo', 'USA Today'],
    'Center': ['Reuters', 'MarketWatch', 'Financial Times', 'Newsweek', 'Forbes'],
    'Lean Right': ['TheDispatch', 'EpochTimes', 'FoxBusiness', 'WSJ', 'National Review', 'WashTimes'],
    'Right': ['Breitbart', 'TheBlaze', 'Daily Mail', 'DailyWire', 'FoxNews', 'NYPost', 'Newsmax'],
}

# ------------------------------ #
#         Helper Functions       #
# ------------------------------ #

def load_jsonl(jsonl_file):
    """
    Loads data from a JSON Lines (JSONL) file into a pandas DataFrame.

    Args:
        jsonl_file (str): Path to the JSONL file.

    Returns:
        pd.DataFrame: DataFrame containing all articles with sentiment/emotion fields.
    """
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL data"):
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
    df = pd.DataFrame(records)
    return df

def map_media_outlet_to_category(df, media_categories):
    """
    Maps each media outlet to its corresponding media category.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        media_categories (dict): Dictionary mapping categories to outlets.

    Returns:
        pd.DataFrame: Updated DataFrame with new 'media_category' column.
    """
    # Create a reverse mapping from outlet to category
    outlet_to_category = {}
    for category, outlets in media_categories.items():
        for outlet in outlets:
            outlet_to_category[outlet.lower().strip()] = category

    # Check if 'media_outlet' field exists
    if 'media_outlet' not in df.columns:
        logging.error("'media_outlet' column not found in DataFrame.")
        raise KeyError("'media_outlet' column not found in DataFrame.")

    # Clean 'media_outlet' field
    df['media_outlet_clean'] = df['media_outlet'].str.lower().str.strip()

    # Map 'media_outlet_clean' to 'media_category'
    df['media_category'] = df['media_outlet_clean'].map(outlet_to_category)

    # Handle unmapped outlets by assigning 'Other'
    df['media_category'] = df['media_category'].fillna('Other')  # Direct assignment to avoid FutureWarning
    unmapped_outlets = df[df['media_category'] == 'Other']['media_outlet'].unique()
    if len(unmapped_outlets) > 0:
        logging.warning(f"Unmapped media outlets found: {unmapped_outlets}")
        print(f"Warning: The following media outlets were not mapped and categorized as 'Other': {unmapped_outlets}")

    return df

def aggregate_sentiment_scores(df, sentiment_categories):
    """
    Aggregates (sums) all sentiment/emotion scores for each category per media category.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        sentiment_categories (list): List of sentiment/emotion categories.

    Returns:
        pd.DataFrame: Aggregated DataFrame with summed scores and counts per media category and sentiment/emotion.
    """
    aggregation = []

    for media_category in MEDIA_CATEGORIES.keys():
        # Filter rows matching the current media category
        mask = (df['media_category'] == media_category)
        subset = df[mask]

        for sentiment in sentiment_categories:
            # Identify all _n fields for the current sentiment
            pattern = f"^{re.escape(sentiment)}_\\d+$"
            matched_cols = [col for col in df.columns if re.match(pattern, col)]
            
            if matched_cols:
                # Identify negative quotation-based scores before clipping
                negative_quot = subset[matched_cols] < 0
                negative_quot_count = negative_quot.sum().sum()
                if negative_quot_count > 0:
                    logging.warning(f"{negative_quot_count} negative quotation-based scores found for sentiment/emotion '{sentiment}' in media category '{media_category}' and set to zero.")
                
                # Sum all quotation-based scores for the current sentiment, setting negatives to zero
                quotation_sum = subset[matched_cols].clip(lower=0).sum(skipna=True).sum()
                # Count the total number of non-NA and non-negative quotation-based scores
                quotation_count = subset[matched_cols].clip(lower=0).count().sum()
            else:
                quotation_sum = 0
                quotation_count = 0
                logging.warning(f"No quotation-based fields found for sentiment/emotion '{sentiment}' in media category '{media_category}'.")

            # Aggregate fulltext-based scores, setting negatives to zero
            fulltext_col = f"{sentiment}_fulltext"
            if fulltext_col in df.columns:
                # Identify negative fulltext-based scores before clipping
                negative_full = subset[fulltext_col] < 0
                negative_full_count = negative_full.sum()
                if negative_full_count > 0:
                    logging.warning(f"{negative_full_count} negative fulltext-based scores found for sentiment/emotion '{sentiment}' in media category '{media_category}' and set to zero.")
                
                fulltext_sum = subset[fulltext_col].clip(lower=0).sum(skipna=True)
                fulltext_count = subset[fulltext_col].clip(lower=0).count()
            else:
                fulltext_sum = 0
                fulltext_count = 0
                logging.warning(f"Fulltext field '{fulltext_col}' not found in DataFrame.")

            aggregation.append({
                'Media Category': media_category,
                'Sentiment/Emotion': sentiment,
                'Quotation_Sum': quotation_sum,
                'Quotation_Count': quotation_count,
                'Fulltext_Sum': fulltext_sum,
                'Fulltext_Count': fulltext_count
            })

    aggregated_df = pd.DataFrame(aggregation)
    return aggregated_df

def calculate_averages(aggregated_df):
    """
    Calculates the average scores for both quotation-based and fulltext sentiment/emotion scores.

    Args:
        aggregated_df (pd.DataFrame): Aggregated DataFrame with summed scores and counts.

    Returns:
        pd.DataFrame: DataFrame containing average scores.
    """
    # Calculate averages, handling division by zero
    aggregated_df['Quotation_Average'] = aggregated_df.apply(
        lambda row: row['Quotation_Sum'] / row['Quotation_Count'] if row['Quotation_Count'] > 0 else None, axis=1
    )
    aggregated_df['Fulltext_Average'] = aggregated_df.apply(
        lambda row: row['Fulltext_Sum'] / row['Fulltext_Count'] if row['Fulltext_Count'] > 0 else None, axis=1
    )

    # Validate that averages are non-negative
    negative_quot_avg = aggregated_df['Quotation_Average'] < 0
    negative_full_avg = aggregated_df['Fulltext_Average'] < 0

    if aggregated_df['Quotation_Average'][negative_quot_avg].any():
        logging.error("Negative values found in Quotation_Average. Setting them to None.")
        aggregated_df.loc[negative_quot_avg, 'Quotation_Average'] = None

    if aggregated_df['Fulltext_Average'][negative_full_avg].any():
        logging.error("Negative values found in Fulltext_Average. Setting them to None.")
        aggregated_df.loc[negative_full_avg, 'Fulltext_Average'] = None

    return aggregated_df

def calculate_mean_median(aggregated_df):
    """
    Calculates the mean and median for each sentiment/emotion category across media categories.

    Args:
        aggregated_df (pd.DataFrame): DataFrame containing average scores.

    Returns:
        pd.DataFrame: DataFrame containing mean and median statistics.
    """
    stats = []

    for sentiment in CATEGORIES:
        sentiment_data = aggregated_df[aggregated_df['Sentiment/Emotion'] == sentiment]

        # Quotation-Based
        quotation_avg = sentiment_data['Quotation_Average'].dropna()
        if not quotation_avg.empty:
            mean_quotation = quotation_avg.mean()
            median_quotation = quotation_avg.median()
        else:
            mean_quotation = None
            median_quotation = None
            logging.warning(f"No quotation-based data available for sentiment/emotion '{sentiment}'.")

        # Fulltext-Based
        fulltext_avg = sentiment_data['Fulltext_Average'].dropna()
        if not fulltext_avg.empty:
            mean_fulltext = fulltext_avg.mean()
            median_fulltext = fulltext_avg.median()
        else:
            mean_fulltext = None
            median_fulltext = None
            logging.warning(f"No fulltext-based data available for sentiment/emotion '{sentiment}'.")

        stats.append({
            'Sentiment/Emotion': sentiment,
            'Mean_Quotation_Average': mean_quotation,
            'Median_Quotation_Average': median_quotation,
            'Mean_Fulltext_Average': mean_fulltext,
            'Median_Fulltext_Average': median_fulltext
        })

    stats_df = pd.DataFrame(stats)
    return stats_df

def save_aggregated_scores_to_csv(aggregated_df, csv_output_dir):
    """
    Saves aggregated scores into a CSV file.

    Args:
        aggregated_df (pd.DataFrame): Aggregated DataFrame with summed scores, counts, and averages.
        csv_output_dir (str): Directory to save the CSV file.

    Returns:
        None
    """
    csv_file = os.path.join(csv_output_dir, "aggregated_sentiment_emotion_scores.csv")
    try:
        aggregated_df.to_csv(csv_file, index=False)
        print(f"Aggregated sentiment/emotion scores saved to '{csv_file}'.")
        logging.info(f"Aggregated sentiment/emotion scores saved to '{csv_file}'.")
    except Exception as e:
        print(f"Error saving aggregated scores to CSV: {e}")
        logging.error(f"Error saving aggregated scores to CSV: {e}")

def plot_statistics(aggregated_df, output_dir):
    """
    Generates 20 bar plots (10 fulltext-based and 10 quotation-based) for each sentiment/emotion category.
    Each plot contains 6 bars representing the six media categories.

    Args:
        aggregated_df (pd.DataFrame): Aggregated DataFrame with summed scores, counts, and averages.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    sns.set_style("whitegrid")

    for sentiment in CATEGORIES:
        # Filter aggregated_df for the current sentiment/emotion
        sentiment_data = aggregated_df[aggregated_df['Sentiment/Emotion'] == sentiment]

        # ----------------------- #
        # Quotation-Based Plot    #
        # ----------------------- #
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Media Category',
            y='Quotation_Average',
            data=sentiment_data,
            palette='Blues_d'
        )
        plt.title(f"Mean Quotation-Based '{sentiment.capitalize()}' Scores Across Media Categories", fontsize=14)
        plt.xlabel('Media Category', fontsize=12)
        plt.ylabel('Mean Quotation-Based Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        quote_plot_filename = f"quote_{sentiment}.png"
        quote_plot_path = os.path.join(output_dir, quote_plot_filename)
        try:
            plt.savefig(quote_plot_path)
            plt.close()
            print(f"Quotation-Based '{sentiment}' scores plot saved to '{quote_plot_path}'.")
            logging.info(f"Quotation-Based '{sentiment}' scores plot saved to '{quote_plot_path}'.")
        except Exception as e:
            print(f"Error saving quotation-based '{sentiment}' scores plot: {e}")
            logging.error(f"Error saving quotation-based '{sentiment}' scores plot: {e}")

        # ----------------------- #
        # Fulltext-Based Plot     #
        # ----------------------- #
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Media Category',
            y='Fulltext_Average',
            data=sentiment_data,
            palette='Oranges_d'
        )
        plt.title(f"Mean Fulltext-Based '{sentiment.capitalize()}' Scores Across Media Categories", fontsize=14)
        plt.xlabel('Media Category', fontsize=12)
        plt.ylabel('Mean Fulltext-Based Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fulltext_plot_filename = f"fulltext_{sentiment}.png"
        fulltext_plot_path = os.path.join(output_dir, fulltext_plot_filename)
        try:
            plt.savefig(fulltext_plot_path)
            plt.close()
            print(f"Fulltext-Based '{sentiment}' scores plot saved to '{fulltext_plot_path}'.")
            logging.info(f"Fulltext-Based '{sentiment}' scores plot saved to '{fulltext_plot_path}'.")
        except Exception as e:
            print(f"Error saving fulltext-based '{sentiment}' scores plot: {e}")
            logging.error(f"Error saving fulltext-based '{sentiment}' scores plot: {e}")

def compile_statistics_to_excel(aggregated_df, stats_df, plots_dir, output_excel_file):
    """
    Compiles the statistics and plots into an Excel file with multiple sheets.

    Args:
        aggregated_df (pd.DataFrame): Aggregated DataFrame with summed scores, counts, and averages.
        stats_df (pd.DataFrame): DataFrame containing mean and median statistics.
        plots_dir (str): Directory containing the plot images.
        output_excel_file (str): Path to the output Excel file.

    Returns:
        None
    """
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        # Write aggregated scores to the first sheet
        aggregated_df.to_excel(writer, sheet_name='Aggregated_Scores', index=False)

        # Write statistics to the second sheet
        stats_df.to_excel(writer, sheet_name='Mean_Median_Statistics', index=False)

        workbook = writer.book

        # Insert plots into separate sheets
        for sentiment in CATEGORIES:
            # Quotation-Based Plot
            quote_plot_filename = f"quote_{sentiment}.png"
            quote_plot_path = os.path.join(plots_dir, quote_plot_filename)
            if os.path.exists(quote_plot_path):
                sheet_title = f"Quote_{sentiment[:28]}"  # Excel sheet name limit
                worksheet = workbook.create_sheet(title=sheet_title)
                try:
                    img = ExcelImage(quote_plot_path)
                    img.anchor = 'A1'
                    worksheet.add_image(img)
                except Exception as e:
                    logging.error(f"Error embedding image '{quote_plot_path}' into Excel: {e}")
            else:
                logging.warning(f"Plot file '{quote_plot_path}' not found. Skipping embedding in Excel.")

            # Fulltext-Based Plot
            fulltext_plot_filename = f"fulltext_{sentiment}.png"
            fulltext_plot_path = os.path.join(plots_dir, fulltext_plot_filename)
            if os.path.exists(fulltext_plot_path):
                sheet_title = f"Fulltext_{sentiment[:25]}"  # Excel sheet name limit
                worksheet = workbook.create_sheet(title=sheet_title)
                try:
                    img = ExcelImage(fulltext_plot_path)
                    img.anchor = 'A1'
                    worksheet.add_image(img)
                except Exception as e:
                    logging.error(f"Error embedding image '{fulltext_plot_path}' into Excel: {e}")
            else:
                logging.warning(f"Plot file '{fulltext_plot_path}' not found. Skipping embedding in Excel.")

        # No need to call writer.save() as it's handled by the 'with' context manager

    print(f"All statistics and plots have been compiled into '{output_excel_file}'.")
    logging.info(f"All statistics and plots have been compiled into '{output_excel_file}'.")

def setup_logging(log_file='analysis.log'):
    """
    Configures the logging settings.

    Args:
        log_file (str): Path to the log file.

    Returns:
        None
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized.")

# ------------------------------ #
#             Main               #
# ------------------------------ #

def main():
    # Setup logging
    setup_logging()

    # Load data
    print("Loading data from JSONL file...")
    try:
        df = load_jsonl(INPUT_JSONL_FILE)
        print(f"Total articles loaded: {len(df)}")
        logging.info(f"Total articles loaded: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return

    if df.empty:
        print("The input file is empty. Exiting.")
        logging.info("The input file is empty. Exiting.")
        return

    # Map media outlets to categories
    print("Mapping media outlets to categories...")
    try:
        df = map_media_outlet_to_category(df, MEDIA_CATEGORIES)
        print("Media outlets mapped to categories successfully.\n")
        logging.info("Media outlets mapped to categories successfully.")
    except Exception as e:
        print(f"Error mapping media outlets to categories: {e}")
        logging.error(f"Error mapping media outlets to categories: {e}")
        return

    # Aggregate sentiment/emotion scores per media category
    print("Aggregating sentiment/emotion scores per media category...")
    try:
        aggregated_df = aggregate_sentiment_scores(df, CATEGORIES)
        aggregated_df = calculate_averages(aggregated_df)
        print("Aggregation of sentiment/emotion scores completed successfully.\n")
        logging.info("Aggregation of sentiment/emotion scores completed successfully.")
    except Exception as e:
        print(f"Error aggregating sentiment/emotion scores: {e}")
        logging.error(f"Error aggregating sentiment/emotion scores: {e}")
        return

    # Calculate mean and median statistics
    print("Calculating mean and median statistics...")
    try:
        stats_df = calculate_mean_median(aggregated_df)
        print("Mean and median statistics calculated successfully.\n")
        logging.info("Mean and median statistics calculated successfully.")
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        logging.error(f"Error calculating statistics: {e}")
        return

    # Save aggregated scores to CSV
    print("Saving aggregated scores to CSV files...")
    try:
        save_aggregated_scores_to_csv(aggregated_df, CSV_OUTPUT_DIR)
        print("Aggregated scores saved to CSV files successfully.\n")
        logging.info("Aggregated scores saved to CSV files successfully.")
    except Exception as e:
        print(f"Error saving aggregated scores to CSV: {e}")
        logging.error(f"Error saving aggregated scores to CSV: {e}")
        return

    # Plot statistics
    print("Generating plots for statistics...")
    try:
        plot_statistics(aggregated_df, OUTPUT_DIR)
        print("Plots generated successfully.\n")
        logging.info("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        logging.error(f"Error generating plots: {e}")
        return

    # Compile statistics and plots into Excel
    print("Compiling statistics and plots into Excel file...")
    try:
        compile_statistics_to_excel(aggregated_df, stats_df, OUTPUT_DIR, OUTPUT_EXCEL_FILE)
        print("Statistics and plots compiled into Excel file successfully.\n")
        logging.info("Statistics and plots compiled into Excel file successfully.")
    except Exception as e:
        print(f"Error compiling statistics and plots into Excel: {e}")
        logging.error(f"Error compiling statistics and plots into Excel: {e}")
        return

    print("\nAnalysis completed successfully.")
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
