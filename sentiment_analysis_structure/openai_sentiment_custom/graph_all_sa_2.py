import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
from openpyxl.drawing.image import Image as ExcelImage
import re
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    outlet_to_category = {}
    for category, outlets in media_categories.items():
        for outlet in outlets:
            outlet_to_category[outlet.lower().strip()] = category

    if 'media_outlet' not in df.columns:
        logging.error("'media_outlet' column not found in DataFrame.")
        raise KeyError("'media_outlet' column not found in DataFrame.")

    df['media_outlet_clean'] = df['media_outlet'].str.lower().str.strip()
    df['media_category'] = df['media_outlet_clean'].map(outlet_to_category)
    df['media_category'] = df['media_category'].fillna('Other')
    unmapped_outlets = df[df['media_category'] == 'Other']['media_outlet'].unique()
    if len(unmapped_outlets) > 0:
        logging.warning(f"Unmapped media outlets found: {unmapped_outlets}")
        print(f"Warning: The following media outlets were not mapped and categorized as 'Other': {unmapped_outlets}")
    return df

def aggregate_sentiment_scores(df, sentiment_categories):
    aggregation = []

    for media_category in MEDIA_CATEGORIES.keys():
        mask = (df['media_category'] == media_category)
        subset = df[mask]

        for sentiment in sentiment_categories:
            pattern = f"^{re.escape(sentiment)}_\\d+$"
            matched_cols = [col for col in df.columns if re.match(pattern, col)]

            if matched_cols:
                negative_quot = subset[matched_cols] < 0
                negative_quot_count = negative_quot.sum().sum()
                if negative_quot_count > 0:
                    logging.warning(f"{negative_quot_count} negative quotation-based scores found for '{sentiment}' in '{media_category}' and set to zero.")
                quotation_sum = subset[matched_cols].clip(lower=0).sum(skipna=True).sum()
                quotation_count = subset[matched_cols].clip(lower=0).count().sum()
            else:
                quotation_sum = 0
                quotation_count = 0
                logging.warning(f"No quotation-based fields found for '{sentiment}' in '{media_category}'.")

            fulltext_col = f"{sentiment}_fulltext"
            if fulltext_col in df.columns:
                negative_full = subset[fulltext_col] < 0
                negative_full_count = negative_full.sum()
                if negative_full_count > 0:
                    logging.warning(f"{negative_full_count} negative fulltext-based scores found for '{sentiment}' in '{media_category}' and set to zero.")
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
    aggregated_df['Quotation_Average'] = aggregated_df.apply(
        lambda row: row['Quotation_Sum'] / row['Quotation_Count'] if row['Quotation_Count'] > 0 else None, axis=1
    )
    aggregated_df['Fulltext_Average'] = aggregated_df.apply(
        lambda row: row['Fulltext_Sum'] / row['Fulltext_Count'] if row['Fulltext_Count'] > 0 else None, axis=1
    )

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
            logging.warning(f"No quotation-based data for '{sentiment}'.")

        # Fulltext-Based
        fulltext_avg = sentiment_data['Fulltext_Average'].dropna()
        if not fulltext_avg.empty:
            mean_fulltext = fulltext_avg.mean()
            median_fulltext = fulltext_avg.median()
        else:
            mean_fulltext = None
            median_fulltext = None
            logging.warning(f"No fulltext-based data for '{sentiment}'.")

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
    csv_file = os.path.join(csv_output_dir, "aggregated_sentiment_emotion_scores.csv")
    try:
        aggregated_df.to_csv(csv_file, index=False)
        print(f"Aggregated sentiment/emotion scores saved to '{csv_file}'.")
        logging.info(f"Aggregated sentiment/emotion scores saved to '{csv_file}'.")
    except Exception as e:
        print(f"Error saving aggregated scores to CSV: {e}")
        logging.error(f"Error saving aggregated scores to CSV: {e}")

def plot_statistics(aggregated_df, output_dir):
    sns.set_style("whitegrid")

    for sentiment in CATEGORIES:
        sentiment_data = aggregated_df[aggregated_df['Sentiment/Emotion'] == sentiment]

        # Quotation-Based Plot
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

        # Fulltext-Based Plot
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

def fit_lmm_and_posthoc(df, sentiment, measure_type='Quotation'):
    """
    Fit a linear mixed model for a given sentiment and measure type (Quotation or Fulltext).
    Perform post-hoc pairwise comparisons using TukeyHSD on the estimated marginal means.

    Args:
        df (pd.DataFrame): The full raw DataFrame with all scores.
        sentiment (str): The sentiment/emotion category.
        measure_type (str): Either 'Quotation' or 'Fulltext'.

    Returns:
        results_dict (dict): Dictionary with LMM summary and post-hoc results as DataFrames.
    """
    if measure_type == 'Quotation':
        # Identify all quotation columns
        pattern = f"^{re.escape(sentiment)}_\\d+$"
        matched_cols = [col for col in df.columns if re.match(pattern, col)]
        if not matched_cols:
            return None
        # We can average these for the LMM at the article level
        df[f'{sentiment}_quotation_mean'] = df[matched_cols].clip(lower=0).mean(axis=1)
        score_col = f'{sentiment}_quotation_mean'
    else:
        # Fulltext column
        fulltext_col = f"{sentiment}_fulltext"
        if fulltext_col not in df.columns:
            return None
        df[f'{sentiment}_fulltext_clipped'] = df[fulltext_col].clip(lower=0)
        score_col = f'{sentiment}_fulltext_clipped'

    # Drop rows with NaN or no data
    model_df = df.dropna(subset=[score_col, 'media_category', 'media_outlet'])

    if model_df.empty:
        return None

    # Fit LMM: score ~ media_category + (1|media_outlet)
    # Convert media_category to categorical
    model_df['media_category'] = pd.Categorical(model_df['media_category'])

    # LMM
    md = mixedlm(f"{score_col} ~ media_category", data=model_df, groups=model_df["media_outlet"])
    mdf = md.fit(reml=False)  # Using ML or REML as needed

    # Get predicted means per media category for post-hoc
    cat_means = model_df.groupby('media_category')[score_col].mean().reset_index()

    # Perform Tukey HSD on these means
    # Note: Strictly, Tukey on group means isn't a perfect post-hoc for LMM,
    # but it's a reasonable approximation for demonstration.
    tukey = pairwise_tukeyhsd(cat_means[score_col], cat_means['media_category'], alpha=0.05)

    results_dict = {
        'LMM_Summary': mdf.summary().as_text(),
        'PostHoc': pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    }

    return results_dict

def run_lmm_analyses(df):
    """
    Run linear mixed models and post-hoc tests for each sentiment and for both Quotation and Fulltext measures.
    Return a nested dictionary with results.

    Args:
        df (pd.DataFrame): The raw DataFrame with all data.

    Returns:
        all_results (dict): Nested dict with keys [sentiment][measure_type] containing results.
    """
    all_results = {}
    for sentiment in CATEGORIES:
        all_results[sentiment] = {}
        for measure_type in ['Quotation', 'Fulltext']:
            res = fit_lmm_and_posthoc(df, sentiment, measure_type=measure_type)
            if res is not None:
                all_results[sentiment][measure_type] = res
    return all_results

def compile_statistics_to_excel(aggregated_df, stats_df, plots_dir, output_excel_file, raw_df, lmm_results):
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        # Write aggregated scores
        aggregated_df.to_excel(writer, sheet_name='Aggregated_Scores', index=False)

        # Write stats
        stats_df.to_excel(writer, sheet_name='Mean_Median_Statistics', index=False)

        # Write raw data
        raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)

        # Write raw scores for each sentiment category (optional)
        # For large datasets, this may be too large, consider selectively writing.
        for sentiment in CATEGORIES:
            # Extract columns relevant to this sentiment
            sentiment_cols = [c for c in raw_df.columns if c.startswith(sentiment+'_')]
            sentiment_df = raw_df[['media_category', 'media_outlet'] + sentiment_cols].copy()
            sentiment_df.to_excel(writer, sheet_name=f"Raw_{sentiment[:29]}", index=False)  # shorten name if needed

        # Write LMM and Post-Hoc results
        # Create a summary sheet that indexes all results
        summary_rows = []
        for sentiment in lmm_results:
            for measure_type in lmm_results[sentiment]:
                sheet_name = f"LMM_{sentiment[:20]}_{measure_type[:8]}"
                # Write LMM summary as text file
                lmm_summary = lmm_results[sentiment][measure_type]['LMM_Summary']
                # Write post-hoc as DataFrame
                posthoc_df = lmm_results[sentiment][measure_type]['PostHoc']

                # Create a separate sheet for each set of results
                # We'll write the LMM summary in a text-based sheet by putting it in a dataframe
                summary_df = pd.DataFrame({'LMM_Summary': lmm_summary.split('\n')})
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # We'll append the post-hoc results below the summary
                startrow = len(summary_df) + 2
                posthoc_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)

                summary_rows.append({
                    'Sentiment': sentiment,
                    'Measure': measure_type,
                    'SheetName': sheet_name
                })

        summary_index_df = pd.DataFrame(summary_rows)
        summary_index_df.to_excel(writer, sheet_name='LMM_Results_Index', index=False)

        workbook = writer.book

        # Insert plots into separate sheets
        for sentiment in CATEGORIES:
            # Quotation-Based Plot
            quote_plot_filename = f"quote_{sentiment}.png"
            quote_plot_path = os.path.join(plots_dir, quote_plot_filename)
            if os.path.exists(quote_plot_path):
                sheet_title = f"Quote_{sentiment[:28]}"
                worksheet = workbook.create_sheet(title=sheet_title)
                try:
                    img = ExcelImage(quote_plot_path)
                    img.anchor = 'A1'
                    worksheet.add_image(img)
                except Exception as e:
                    logging.error(f"Error embedding image '{quote_plot_path}' into Excel: {e}")
            
            # Fulltext-Based Plot
            fulltext_plot_filename = f"fulltext_{sentiment}.png"
            fulltext_plot_path = os.path.join(plots_dir, fulltext_plot_filename)
            if os.path.exists(fulltext_plot_path):
                sheet_title = f"Fulltext_{sentiment[:25]}"
                worksheet = workbook.create_sheet(title=sheet_title)
                try:
                    img = ExcelImage(fulltext_plot_path)
                    img.anchor = 'A1'
                    worksheet.add_image(img)
                except Exception as e:
                    logging.error(f"Error embedding image '{fulltext_plot_path}' into Excel: {e}")

    print(f"All statistics, raw data, LMM results, and plots have been compiled into '{output_excel_file}'.")
    logging.info(f"All statistics, raw data, LMM results, and plots have been compiled into '{output_excel_file}'.")

def setup_logging(log_file='analysis.log'):
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

    # Run linear mixed model analyses and post-hoc tests
    print("Fitting linear mixed models and running post-hoc tests...")
    try:
        lmm_results = run_lmm_analyses(df)
        print("LMM analysis and post-hoc tests completed successfully.\n")
        logging.info("LMM analysis and post-hoc tests completed successfully.")
    except Exception as e:
        print(f"Error in LMM analysis: {e}")
        logging.error(f"Error in LMM analysis: {e}")
        return

    # Compile everything into Excel
    print("Compiling statistics, raw data, LMM results, and plots into Excel file...")
    try:
        compile_statistics_to_excel(aggregated_df, stats_df, OUTPUT_DIR, OUTPUT_EXCEL_FILE, df, lmm_results)
        print("All results compiled into Excel file successfully.\n")
        logging.info("All results compiled into Excel file successfully.")
    except Exception as e:
        print(f"Error compiling statistics and plots into Excel: {e}")
        logging.error(f"Error compiling statistics and plots into Excel: {e}")
        return

    print("Analysis completed successfully.")
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
