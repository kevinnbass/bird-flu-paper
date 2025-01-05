import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
from openpyxl.drawing.image import Image as ExcelImage

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Input JSONL file produced by the first script
INPUT_JSONL_FILE = 'sentiment_emotion_analysis_openai.jsonl'

# Output directory for the graphs
OUTPUT_DIR = 'graphs_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output Excel file
OUTPUT_EXCEL_FILE = 'analysis_results.xlsx'

# Define the ten sentiment/emotion categories
CATEGORIES = [
    'joy', 'sadness', 'anger', 'fear',
    'surprise', 'disgust', 'trust', 'anticipation',
    'negative_sentiment', 'positive_sentiment'
]

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
        for line in f:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
    df = pd.DataFrame(records)
    return df

def calculate_statistics(df, categories):
    """
    Calculates mean and median for each specified category.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        categories (list): List of category names to analyze.

    Returns:
        pd.DataFrame: DataFrame with mean and median for each category.
    """
    stats = []
    for category in categories:
        if category in df.columns:
            mean_val = df[category].mean()
            median_val = df[category].median()
        else:
            logging.warning(f"Category '{category}' not found in DataFrame.")
            mean_val = None
            median_val = None
        stats.append({
            'Category': category,
            'Mean': mean_val,
            'Median': median_val
        })
    stats_df = pd.DataFrame(stats)
    return stats_df

def plot_statistics(stats_df, output_dir):
    """
    Generates bar plots for mean and median of each category.

    Args:
        stats_df (pd.DataFrame): DataFrame containing mean and median statistics.
        output_dir (str): Directory to save the plots.

    Returns:
        None
    """
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Plot Mean Values
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Category', y='Mean', data=stats_df, palette='viridis')
    plt.title('Mean Values of Sentiment and Emotion Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Mean Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    mean_plot_path = os.path.join(output_dir, 'mean_values.png')
    plt.savefig(mean_plot_path)
    plt.close()
    print(f"Mean values plot saved as {mean_plot_path}.")
    logging.info(f"Mean values plot saved as {mean_plot_path}.")

    # Plot Median Values
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Category', y='Median', data=stats_df, palette='magma')
    plt.title('Median Values of Sentiment and Emotion Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Median Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    median_plot_path = os.path.join(output_dir, 'median_values.png')
    plt.savefig(median_plot_path)
    plt.close()
    print(f"Median values plot saved as {median_plot_path}.")
    logging.info(f"Median values plot saved as {median_plot_path}.")

def save_to_excel(stats_df, output_excel_file, plots_dir):
    """
    Saves the statistics and plots into an Excel file with multiple sheets.

    Args:
        stats_df (pd.DataFrame): DataFrame containing statistics.
        output_excel_file (str): Path to the output Excel file.
        plots_dir (str): Directory containing the plot images.

    Returns:
        None
    """
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        # Write statistics to the first sheet
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        workbook = writer.book

        # Insert Mean Plot
        mean_sheet = workbook.create_sheet(title='Mean Plot')
        mean_image_path = os.path.join(plots_dir, 'mean_values.png')
        if os.path.exists(mean_image_path):
            mean_img = ExcelImage(mean_image_path)
            mean_sheet.add_image(mean_img, 'A1')
        else:
            logging.warning(f"Mean plot image not found at {mean_image_path}.")

        # Insert Median Plot
        median_sheet = workbook.create_sheet(title='Median Plot')
        median_image_path = os.path.join(plots_dir, 'median_values.png')
        if os.path.exists(median_image_path):
            median_img = ExcelImage(median_image_path)
            median_sheet.add_image(median_img, 'A1')
        else:
            logging.warning(f"Median plot image not found at {median_image_path}.")

        # Save the Excel file
        writer.save()
    print(f"All results saved to {output_excel_file}.")
    logging.info(f"All results saved to {output_excel_file}.")

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

    # Exclude 'fear' field as per your instruction
    # If 'sentiment' is present and needs to be excluded, handle accordingly
    categories_to_analyze = [cat for cat in CATEGORIES if cat != 'fear']

    # Calculate statistics
    print("Calculating mean and median for each category...")
    try:
        stats_df = calculate_statistics(df, categories_to_analyze)
        print("Statistics calculated successfully.\n")
        logging.info("Statistics calculated successfully.")
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        logging.error(f"Error calculating statistics: {e}")
        return

    # Display statistics
    print("Mean and Median Statistics:")
    print(stats_df)
    logging.info("Mean and Median Statistics:")
    logging.info(f"\n{stats_df}")

    # Generate plots
    print("Generating plots...")
    try:
        plot_statistics(stats_df, OUTPUT_DIR)
        logging.info("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        logging.error(f"Error generating plots: {e}")
        return

    # Save results to Excel
    print("Saving results to Excel...")
    try:
        save_to_excel(stats_df, OUTPUT_EXCEL_FILE, OUTPUT_DIR)
        logging.info("Results saved to Excel successfully.")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        logging.error(f"Error saving to Excel: {e}")
        return

    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()
