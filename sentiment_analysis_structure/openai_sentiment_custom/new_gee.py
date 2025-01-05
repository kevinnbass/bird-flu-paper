import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from tqdm import tqdm
import re
import sys
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson, NegativeBinomial
from statsmodels.genmod.cov_struct import Unstructured
from scipy import stats  # for Q-Q plots

# Configuration
INPUT_JSONL_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'
os.makedirs('graphs_analysis', exist_ok=True)

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

def load_jsonl(jsonl_file):
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

def compute_representative_measure(df):
    # Just use 'joy' as an example for demonstration
    sentiment = 'joy'
    pattern = f"^{re.escape(sentiment)}_\\d+$"
    matched_cols = [col for col in df.columns if re.match(pattern, col)]
    if matched_cols:
        df['joy_quotation_mean'] = df[matched_cols].clip(lower=0).mean(axis=1)
        return 'joy_quotation_mean'
    else:
        # fallback
        for s in CATEGORIES:
            pat = f"^{re.escape(s)}_\\d+$"
            mc = [c for c in df.columns if re.match(pat, c)]
            if mc:
                colname = f'{s}_quotation_mean'
                df[colname] = df[mc].clip(lower=0).mean(axis=1)
                return colname
    return None

def fit_gee(df, score_col, family, cov_struct=Unstructured()):
    df = df.dropna(subset=[score_col, 'media_category', 'media_outlet']).copy()
    if df['media_category'].nunique() < 2:
        return None
    df['media_category'] = df['media_category'].astype('category')
    formula = f"{score_col} ~ media_category"
    model = GEE.from_formula(formula, groups="media_outlet", data=df, family=family, cov_struct=cov_struct)
    results = model.fit()
    return results, df

def plot_residuals(results, model_name):
    resid = results.resid_pearson
    # Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(resid, kde=True, color='blue')
    plt.title(f"Residual Distribution - {model_name}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Q-Q plot
    plt.figure(figsize=(8,5))
    sm.qqplot(resid, line='45', fit=True)
    plt.title(f"Q-Q Plot of Residuals - {model_name}")
    plt.tight_layout()
    plt.show()

def poisson_quasi_likelihood(y, mu):
    # QL for Poisson: sum(y_i * log(mu_i) - mu_i), with convention 0*log(0)=0
    # If mu_i=0, we avoid log(0) by skipping or adding a small value.
    # Typically, mu should never be zero if data >0. Add epsilon if needed.
    epsilon = 1e-10
    mu_clipped = np.maximum(mu, epsilon)
    ql = np.sum(y * np.log(mu_clipped) - mu_clipped)
    return ql

def compute_qic(results, df, score_col, family_str):
    # Compute QIC only for Poisson as demonstration
    if family_str.lower() != 'poisson':
        print("QIC computation for Non-Poisson family is not implemented here.")
        return np.nan

    # Extract needed components
    # robust covariance (Vr)
    Vr = results.cov_params()
    # naive covariance (Vi)
    Vi = results.naive_cov

    # Get fitted means
    mu = results.fittedvalues
    y = df[score_col].values

    # Compute Quasi-likelihood for Poisson
    QL = poisson_quasi_likelihood(y, mu)

    # trace term
    trace_term = np.trace(np.dot(Vr, np.linalg.pinv(Vi)))

    QIC_value = -2 * QL + 2 * trace_term
    return QIC_value

def main():
    df = load_jsonl(INPUT_JSONL_FILE)
    df = map_media_outlet_to_category(df, MEDIA_CATEGORIES)
    score_col = compute_representative_measure(df)
    if score_col is None:
        print("No representative measure found. Cannot proceed.")
        return

    # Fit GEE Poisson
    print("Fitting GEE with Poisson...")
    poisson_results, poisson_df = fit_gee(df, score_col, Poisson())
    if poisson_results is not None:
        print(poisson_results.summary())
        plot_residuals(poisson_results, "GEE-Poisson")
        QIC_poisson = compute_qic(poisson_results, poisson_df, score_col, 'poisson')
        print(f"Poisson QIC: {QIC_poisson}")

    # Fit GEE Negative Binomial
    print("Fitting GEE with Negative Binomial...")
    nb_results, nb_df = fit_gee(df, score_col, NegativeBinomial())
    if nb_results is not None:
        print(nb_results.summary())
        plot_residuals(nb_results, "GEE-Negative Binomial")
        # QIC for NB not implemented here
        print("QIC computation for Negative Binomial not implemented. Compare QIC from Poisson or consider QIC approximations.\n")

    # Discussion:
    # Residual plots give a heuristic sense of model fit. If Negative Binomial residuals look "better" (less skewed),
    # it might indicate NB is more appropriate.
    # But for a rigorous approach, QIC is more reliable for GEE. We computed QIC for Poisson as example.
    # For NB, you'd need a suitable quasi-likelihood expression or another approach.
    # Consider QIC as a primary tool to guide model selection, and residuals as a secondary heuristic.

if __name__ == "__main__":
    main()
