import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
from tqdm import tqdm
from openpyxl.drawing.image import Image as ExcelImage
import re
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import numpy as np
import patsy

import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable, Independence, Unstructured, Autoregressive
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm, pearsonr

# ------------------------------------------ #
#   Remove excessive FontManager debugging   #
# ------------------------------------------ #
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# ------------------------------------------ #
#       Configuration and Globals           #
# ------------------------------------------ #

INPUT_JSONL_FILE = 'processed_all_articles_merged.jsonl'
OUTPUT_DIR = 'graphs_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUTPUT_DIR = 'csv_raw_scores'
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

OUTPUT_EXCEL_MAIN = 'analysis_main.xlsx'
OUTPUT_EXCEL_RAW = 'analysis_raw.xlsx'
OUTPUT_EXCEL_LMM = 'analysis_gee.xlsx'
OUTPUT_EXCEL_PLOTS = 'analysis_plots.xlsx'
OUTPUT_EXCEL_COMBINED = 'analysis_combined.xlsx'
OUTPUT_EXCEL_GRAPHING = 'analysis_for_graphing.xlsx'
OUTPUT_EXCEL_CORRELATION = 'analysis_correlation_data.xlsx'
OUTPUT_EXCEL_COMBINED_ALL = 'analysis_combined_all.xlsx'
OUTPUT_EXCEL_QIC = 'analysis_gee_qic.xlsx'

CATEGORIES = [
    'joy', 'sadness', 'anger', 'fear',
    'surprise', 'disgust', 'trust', 'anticipation',
    'negative_sentiment', 'positive_sentiment'
]

MEDIA_CATEGORIES = {
    'Scientific': ['nature', 'sciam', 'stat', 'newscientist'],
    'Left': ['theatlantic', 'the daily beast', 'the intercept', 'mother jones', 'msnbc', 'slate', 'vox', 'huffpost'],
    'Lean Left': ['ap', 'axios', 'cnn', 'guardian', 'business insider', 'nbcnews', 'npr', 'nytimes', 'politico', 'propublica', 'wapo', 'usa today'],
    'Center': ['reuters', 'marketwatch', 'financial times', 'newsweek', 'forbes'],
    'Lean Right': ['thedispatch', 'epochtimes', 'foxbusiness', 'wsj', 'national review', 'washtimes'],
    'Right': ['breitbart', 'theblaze', 'daily mail', 'dailywire', 'foxnews', 'nypost', 'newsmax'],
}

CATEGORY_ORDER = ["Scientific", "Left", "Lean Left", "Center", "Lean Right", "Right"]


def setup_logging(log_file='analysis.log'):
    """
    Sets up file + console logging for debugging.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(module)s - %(message)s'
    )
    # Also log to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(module)s: %(message)s'))
    logging.getLogger('').addHandler(console)

    logging.info("Logging initialized (file + console).")


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
    logging.info("Mapping media outlets to media_category.")
    if 'media_outlet' not in df.columns:
        raise KeyError("'media_outlet' column not found in DataFrame!")

    outlet_to_category = {}
    for category, outlets in media_categories.items():
        for outlet in outlets:
            outlet_to_category[outlet.lower().strip()] = category

    df['media_outlet_clean'] = df['media_outlet'].str.lower().str.strip()
    df['media_category'] = df['media_outlet_clean'].map(outlet_to_category)
    df['media_category'] = df['media_category'].fillna('Other')
    # Warn about unmapped
    unmapped = df.loc[df['media_category']=='Other','media_outlet'].unique()
    if len(unmapped)>0:
        logging.warning(f"Unmapped outlets: {unmapped}")
        print(f"Warning: Some media_outlets mapped to 'Other': {unmapped}")
    return df


def compute_article_level_scores(df, sentiment_categories):
    logging.info("Computing article-level scores for Quotation & Fulltext.")
    for sentiment in sentiment_categories:
        pattern = f"^{re.escape(sentiment)}_\\d+$"
        cols = [c for c in df.columns if re.match(pattern,c)]
        if cols:
            df[f"{sentiment}_quotation_mean_article"] = df[cols].clip(lower=0).mean(axis=1)
        else:
            df[f"{sentiment}_quotation_mean_article"] = np.nan

        fcol = f"{sentiment}_fulltext"
        if fcol in df.columns:
            df[f"{sentiment}_fulltext_article"] = df[fcol].clip(lower=0)
        else:
            df[f"{sentiment}_fulltext_article"] = np.nan
    return df


def aggregate_sentiment_scores(df, sentiment_categories):
    logging.info("Aggregating sentiment/emotion scores per media_category.")
    rows=[]
    for cat in MEDIA_CATEGORIES.keys():
        subset = df[df['media_category']==cat].copy()
        for sentiment in sentiment_categories:
            pattern = f"^{re.escape(sentiment)}_\\d+$"
            matched = [c for c in df.columns if re.match(pattern,c)]
            if matched:
                qsum = subset[matched].clip(lower=0).sum(skipna=True).sum()
                qcount = subset[matched].clip(lower=0).count().sum()
            else:
                qsum,qcount=0,0
            fcol = f"{sentiment}_fulltext"
            if fcol in df.columns:
                fsum = subset[fcol].clip(lower=0).sum(skipna=True)
                fcount = subset[fcol].clip(lower=0).count()
            else:
                fsum,fcount=0,0
            rows.append({
                'Media Category':cat,
                'Sentiment/Emotion':sentiment,
                'Quotation_Sum':qsum,
                'Quotation_Count':qcount,
                'Fulltext_Sum':fsum,
                'Fulltext_Count':fcount
            })
    return pd.DataFrame(rows)


def calculate_averages(aggdf):
    logging.info("Calculating Quotation/Fulltext averages.")
    aggdf['Quotation_Average'] = aggdf.apply(
        lambda r: r['Quotation_Sum']/r['Quotation_Count'] if r['Quotation_Count']>0 else None, axis=1
    )
    aggdf['Fulltext_Average'] = aggdf.apply(
        lambda r: r['Fulltext_Sum']/r['Fulltext_Count'] if r['Fulltext_Count']>0 else None, axis=1
    )
    return aggdf


def calculate_mean_median(aggdf):
    logging.info("Computing overall mean scores across categories, by sentiment.")
    stats=[]
    for sentiment in CATEGORIES:
        subdf = aggdf[aggdf['Sentiment/Emotion']==sentiment].copy()
        qa = subdf['Quotation_Average'].dropna()
        fa = subdf['Fulltext_Average'].dropna()
        stats.append({
            'Sentiment/Emotion':sentiment,
            'Mean_Quotation_Average': qa.mean() if len(qa)>0 else None,
            'Mean_Fulltext_Average': fa.mean() if len(fa)>0 else None
        })
    return pd.DataFrame(stats)


def save_aggregated_scores_to_csv(aggdf, csvoutdir, prefix='aggregated_sentiment_emotion_scores.csv'):
    outp = os.path.join(csvoutdir,prefix)
    try:
        aggdf.to_csv(outp,index=False)
        msg=f"Aggregated sentiment/emotion scores saved to '{outp}'."
        print(msg)
        logging.info(msg)
    except Exception as e:
        logging.error(f"Error saving aggregated: {e}")


def plot_statistics(aggdf, outdir):
    sns.set_style('whitegrid')
    for sentiment in CATEGORIES:
        subdf = aggdf[aggdf['Sentiment/Emotion']==sentiment].copy()

        # Quotation
        plt.figure(figsize=(10,6))
        sns.barplot(x='Media Category',y='Quotation_Average',data=subdf,color='steelblue')
        plt.title(f"Mean Quotation-Based '{sentiment.capitalize()}' Scores")
        plt.xlabel('Media Category')
        plt.ylabel('Mean Quotation-Based Average Score')
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        quote_path = os.path.join(outdir,f"quote_{sentiment}.png")
        try:
            plt.savefig(quote_path)
            plt.close()
            print(f"Quotation-Based '{sentiment}' scores plot saved to '{quote_path}'.")
        except Exception as e:
            logging.error(f"Error saving quotation plot for {sentiment}: {e}")

        # Fulltext
        plt.figure(figsize=(10,6))
        sns.barplot(x='Media Category',y='Fulltext_Average',data=subdf,color='darkorange')
        plt.title(f"Mean Fulltext-Based '{sentiment.capitalize()}' Scores")
        plt.xlabel('Media Category')
        plt.ylabel('Mean Fulltext-Based Average Score')
        plt.xticks(rotation=45,ha='right')
        plt.tight_layout()
        fulltext_path = os.path.join(outdir,f"fulltext_{sentiment}.png")
        try:
            plt.savefig(fulltext_path)
            plt.close()
            print(f"Fulltext-Based '{sentiment}' scores plot saved to '{fulltext_path}'.")
        except Exception as e:
            logging.error(f"Error saving fulltext plot for {sentiment}: {e}")


def main():
    setup_logging()
    logging.info("Starting main script (no code for AR(1) included here).")

    print("Single run, with less font-manager debugging, and fix for Excel writer.")
    df = load_jsonl(INPUT_JSONL_FILE)
    print(f"Total articles loaded: {len(df)}")
    df = map_media_outlet_to_category(df, MEDIA_CATEGORIES)

    # If date is present, convert
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d',errors='coerce')
        logging.info("Converted 'date' column to datetime.")
    else:
        logging.info("'date' column not found, skipping AR(1) scenario if any.")

    # Compute
    df = compute_article_level_scores(df, CATEGORIES)

    # chunk raw
    chunk_size=20000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        cpath = os.path.join(CSV_OUTPUT_DIR,f"raw_data_part_{i//chunk_size+1}.csv")
        chunk.to_csv(cpath,index=False)
        print(f"Saved chunk {i//chunk_size+1} to {cpath}")

    print("\nSummary Statistics:")
    print("Total number of articles:", len(df))
    print("\nNumber of articles per outlet:")
    print(df['media_outlet_clean'].value_counts())
    print("\nNumber of articles per category:")
    print(df['media_category'].value_counts())
    print()

    # Aggregation
    aggdf = aggregate_sentiment_scores(df, CATEGORIES)
    aggdf = calculate_averages(aggdf)
    statsdf = calculate_mean_median(aggdf)
    save_aggregated_scores_to_csv(aggdf, CSV_OUTPUT_DIR)

    plot_statistics(aggdf, OUTPUT_DIR)

    # Simple correlation analysis for Quotation vs Fulltext
    correlation_rows=[]
    corr_scat_data={}
    for sentiment in CATEGORIES:
        subd = aggdf.loc[aggdf['Sentiment/Emotion']==sentiment].copy()
        subd = subd.loc[subd['Media Category'].isin(MEDIA_CATEGORIES.keys())]
        subd.dropna(subset=['Quotation_Average','Fulltext_Average'],inplace=True)
        if len(subd)>1:
            cval,_= pearsonr(subd['Quotation_Average'],subd['Fulltext_Average'])
        else:
            cval=np.nan
        correlation_rows.append({'Sentiment/Emotion':sentiment,'Correlation':cval})
        corr_scat_data[sentiment]= subd[['Media Category','Quotation_Average','Fulltext_Average']].copy()

        plt.figure(figsize=(6,6))
        sns.scatterplot(x='Quotation_Average',y='Fulltext_Average', data=subd, hue='Media Category', s=50)
        plt.title(f"Scatter: {sentiment.capitalize()} (Quotation vs Fulltext)")
        plt.xlabel('Quotation_Average')
        plt.ylabel('Fulltext_Average')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        out_scatter = os.path.join(OUTPUT_DIR, f"scatter_{sentiment}.png")
        plt.savefig(out_scatter)
        plt.close()
        print(f"Scatter plot for {sentiment} saved to '{out_scatter}'.")

    correlation_df= pd.DataFrame(correlation_rows)
    plt.figure(figsize=(10,6))
    sns.barplot(x='Sentiment/Emotion', y='Correlation', data=correlation_df, color='gray')
    plt.title("Correlation Between Quotation and Fulltext Averages")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1,1)
    plt.tight_layout()
    cplot_path = os.path.join(OUTPUT_DIR,"correlation_quotation_fulltext.png")
    plt.savefig(cplot_path)
    plt.close()
    print(f"Correlation plot saved to '{cplot_path}'.")

    # -------------------------- #
    # Fix: use with context here
    # -------------------------- #
    with pd.ExcelWriter(OUTPUT_EXCEL_CORRELATION, engine='openpyxl') as corr_writer:
        correlation_df.to_excel(corr_writer, sheet_name='Correlation', index=False)
        for sentiment, data_ in corr_scat_data.items():
            data_.to_excel(corr_writer, sheet_name=f"{sentiment}_Data", index=False)

    # Combined scatter
    big_list=[]
    for sentiment,data_ in corr_scat_data.items():
        dtmp= data_.copy()
        dtmp['Sentiment/Emotion']=sentiment
        big_list.append(dtmp)
    combined_all= pd.concat(big_list, ignore_index=True)

    combined_all['Quotation_Z']=combined_all.groupby('Sentiment/Emotion')['Quotation_Average'].transform(
        lambda x:(x-x.mean())/x.std() if x.std()!=0 else x - x.mean()
    )
    combined_all['Fulltext_Z']=combined_all.groupby('Sentiment/Emotion')['Fulltext_Average'].transform(
        lambda x:(x-x.mean())/x.std() if x.std()!=0 else x - x.mean()
    )
    combined_all.to_excel(OUTPUT_EXCEL_COMBINED_ALL,index=False)

    comb_no_nan= combined_all.dropna(subset=['Quotation_Z','Fulltext_Z'])
    if len(comb_no_nan)>1:
        r_val,_=pearsonr(comb_no_nan['Quotation_Z'],comb_no_nan['Fulltext_Z'])
    else:
        r_val=np.nan

    plt.figure(figsize=(10,8))
    sns.regplot(x='Quotation_Z', y='Fulltext_Z', data=comb_no_nan, scatter_kws={'color':'black'}, line_kws={'color':'red'})
    plt.title(f"All Sentiments Combined (Normalized)\nR = {r_val:.2f}")
    plt.xlabel("Normalized Quotation_Average (Z-score)")
    plt.ylabel("Normalized Fulltext_Average (Z-score)")
    plt.tight_layout()
    comb_scat_out= os.path.join(OUTPUT_DIR,"combined_normalized_scatter.png")
    plt.savefig(comb_scat_out)
    plt.close()
    print(f"Combined normalized scatter plot saved to '{comb_scat_out}'.")

    # (Placeholder) GEE steps if you want them - not included in detail here
    # ...
    # e.g. gee_results, qic_df = run_gee_analyses(df)

    print("Analysis completed successfully.")


if __name__=="__main__":
    setup_logging()
    main()
