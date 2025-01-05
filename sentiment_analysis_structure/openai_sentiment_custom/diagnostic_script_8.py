import json
import pandas as pd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import numpy as np
from tqdm import tqdm
import logging
import os

INPUT_JSONL_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'
MEDIA_CATEGORIES = {
    'Scientific': ['nature', 'sciam', 'stat', 'newscientist'],
    'Left': ['theatlantic', 'the daily beast', 'the intercept', 'mother jones', 'msnbc', 'slate', 'vox', 'huffpost'],
    'Lean Left': ['ap', 'axios', 'cnn', 'guardian', 'business insider', 'nbcnews', 'npr', 'nytimes', 'politico', 'propublica', 'wapo', 'usa today'],
    'Center': ['reuters', 'marketwatch', 'financial times', 'newsweek', 'forbes'],
    'Lean Right': ['thedispatch', 'epochtimes', 'foxbusiness', 'wsj', 'national review', 'washtimes'],
    'Right': ['breitbart', 'theblaze', 'daily mail', 'dailywire', 'foxnews', 'nypost', 'newsmax'],
}

CATEGORIES = [
    'joy', 'sadness', 'anger', 'fear',
    'surprise', 'disgust', 'trust', 'anticipation',
    'negative_sentiment', 'positive_sentiment'
]

def load_data(jsonl_file):
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL data"):
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
    df = pd.DataFrame(records)
    return df

def map_media_outlet_to_category(df, media_categories):
    outlet_to_category = {}
    for category, outlets in media_categories.items():
        for outlet in outlets:
            outlet_to_category[outlet.lower().strip()] = category

    df['media_outlet_clean'] = df['media_outlet'].str.lower().str.strip()
    df['media_category'] = df['media_outlet_clean'].map(outlet_to_category)
    df['media_category'] = df['media_category'].fillna('Other')
    return df

def filter_low_obs_outlets(df, min_obs=2):
    outlet_counts = df['media_outlet'].value_counts()
    low_outlets = outlet_counts[outlet_counts < min_obs].index.tolist()
    if low_outlets:
        print("Removing outlets with fewer than 2 observations:")
        for o in low_outlets:
            print(f" - {o}")
        df = df[~df['media_outlet'].isin(low_outlets)]
    return df

def compute_representative_measure(df):
    # Try joy_quotation_mean as a representative measure
    sentiment = 'joy'
    pattern = f"^{re.escape(sentiment)}_\\d+$"
    matched_cols = [col for col in df.columns if re.match(pattern, col)]

    if matched_cols:
        df['joy_quotation_mean'] = df[matched_cols].clip(lower=0).mean(axis=1)
        return 'joy_quotation_mean'
    else:
        # Fallback: try another sentiment
        for s in CATEGORIES:
            pat = f"^{re.escape(s)}_\\d+$"
            mc = [c for c in df.columns if re.match(pat, c)]
            if mc:
                colname = f'{s}_quotation_mean'
                df[colname] = df[mc].clip(lower=0).mean(axis=1)
                return colname
        # If none found, return None
        return None

def remove_low_variance_outlets(df, score_col, threshold=0.01):
    # Identify outlets with near-zero variance in the chosen measure
    outlet_var = df.groupby('media_outlet')[score_col].std()
    low_var_outlets = outlet_var[outlet_var.fillna(0) < threshold].index.tolist()

    if low_var_outlets:
        print(f"Removing {len(low_var_outlets)} outlets due to near-zero variance in {score_col} (threshold={threshold}):")
        for o in low_var_outlets:
            print(f" - {o}")
        df = df[~df['media_outlet'].isin(low_var_outlets)]
    else:
        print(f"No outlets removed based on near-zero variance criteria (threshold={threshold}).")
    return df

def try_fit_model(df, score_col):
    # Reset index so that X.index and df.index align nicely
    df = df.reset_index(drop=True)

    # Construct design matrices for debugging
    formula = f"{score_col} ~ media_category"
    y, X = dmatrices(formula, data=df, return_type='dataframe')
    X_rank = np.linalg.matrix_rank(X)
    print(f"[DEBUG] X design matrix shape: {X.shape}")
    print(f"[DEBUG] y vector shape: {y.shape}")
    print(f"[DEBUG] Rank of X: {X_rank}")

    # Align df with X's indices (both should have the same indexing after reset)
    df_aligned = df.loc[X.index]

    try:
        md = smf.mixedlm(formula, data=df_aligned, groups=df_aligned["media_outlet"])
        mdf = md.fit(reml=True, method='lbfgs')
        if not mdf.converged:
            print("Model did not converge.")
        else:
            print("Model converged successfully.")
    except np.linalg.LinAlgError as e:
        print(f"Model failed with a linear algebra error (likely singular): {e}")
    except Exception as e:
        print(f"Model failed: {e}")

def main():
    df = load_data(INPUT_JSONL_FILE)
    print(f"Total articles: {len(df)}")

    df = map_media_outlet_to_category(df, MEDIA_CATEGORIES)
    df = filter_low_obs_outlets(df)

    measure_col = compute_representative_measure(df)
    if measure_col is None:
        print("No representative measure found. Cannot proceed with variability checks.")
        return

    print(f"Using {measure_col} as representative measure for variability checks.")

    # Increase threshold to 0.01
    df = remove_low_variance_outlets(df, measure_col, threshold=0.9)

    print("Attempting to fit a simple LMM with the representative measure...")
    try_fit_model(df, measure_col)

    print("\nIf the model is singular, consider:")
    print("- Removing or merging categories with zero or near-zero variance.")
    print("- Simplifying the model (e.g., remove random effects).")
    print("- Checking if certain outlets always have identical scores.")
    print("- Adjusting the threshold or removing these problematic outlets or categories identified above.")

if __name__ == "__main__":
    main()
