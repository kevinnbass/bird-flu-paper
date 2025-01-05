import json
import pandas as pd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis

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
    # Reset index after filtering
    df = df.reset_index(drop=True)
    return df

def compute_representative_measure(df):
    # Try joy_quotation_mean first
    sentiment = 'joy'
    pattern = f"^{re.escape(sentiment)}_\\d+$"
    matched_cols = [col for col in df.columns if re.match(pattern, col)]

    if matched_cols:
        df['joy_quotation_mean'] = df[matched_cols].clip(lower=0).mean(axis=1)
        return 'joy_quotation_mean'
    else:
        # Fallback to another sentiment
        for s in CATEGORIES:
            pat = f"^{re.escape(s)}_\\d+$"
            mc = [c for c in df.columns if re.match(pat, c)]
            if mc:
                colname = f'{s}_quotation_mean'
                df[colname] = df[mc].clip(lower=0).mean(axis=1)
                return colname
        return None

def check_variability(df, score_col, threshold=1e-6):
    outlet_var = df.groupby('media_outlet')[score_col].std()
    zero_var_outlets = outlet_var[outlet_var.fillna(0) <= threshold].index.tolist()

    if zero_var_outlets:
        print(f"Outlets with near-zero variance (<= {threshold}) in {score_col}:")
        for o in zero_var_outlets:
            print(f" - {o}")
    else:
        print(f"No outlets with near-zero variance in {score_col}.")

def try_fit_model(df, score_col):
    if 'media_category' not in df or 'media_outlet' not in df:
        print("Missing required columns for LMM.")
        return

    if df['media_category'].nunique() < 2:
        print("Not enough categories for LMM.")
        return

    formula = f"{score_col} ~ media_category"

    y, X = dmatrices(formula, data=df, return_type='dataframe')
    X_rank = np.linalg.matrix_rank(X)
    print(f"[DEBUG] X design matrix shape: {X.shape}")
    print(f"[DEBUG] y vector shape: {y.shape}")
    print(f"[DEBUG] Rank of X: {X_rank}")

    try:
        md = smf.mixedlm(formula, data=df, groups=df["media_outlet"])
        mdf = md.fit(reml=True, method='lbfgs')
        if not mdf.converged:
            print("Model did not converge.")
        else:
            print("Model converged successfully.")
    except np.linalg.LinAlgError as e:
        print(f"Model failed with a linear algebra error (likely singular): {e}")
    except Exception as e:
        print(f"Model failed: {e}")

def evaluate_distribution(df, score_col):
    # Basic descriptive statistics
    scores = df[score_col].dropna()
    mean_val = scores.mean()
    var_val = scores.var()
    skew_val = skew(scores)
    kurt_val = kurtosis(scores, fisher=True)

    print("\nDistribution Evaluation:")
    print(f"Mean: {mean_val}")
    print(f"Variance: {var_val}")
    print(f"Skewness: {skew_val}")
    print(f"Kurtosis (excess): {kurt_val}")

    # Frequency table
    freq_table = scores.value_counts().sort_index()
    print("\nFrequency table of the representative measure:")
    print(freq_table)

    # Plotting Histogram and Q-Q Plot
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(scores, kde=False, bins=10, color='steelblue')
    plt.title("Histogram of Representative Measure")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    sm.qqplot(scores, line='s', fit=True)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk test (if sample size <= 5000)
    if len(scores) <= 5000:
        stat, p = shapiro(scores)
        print("\nShapiro-Wilk Test for Normality:")
        print(f"Statistic: {stat}, p-value: {p}")
        if p < 0.05:
            print("Data likely not normally distributed (reject H0).")
        else:
            print("No strong evidence against normality (fail to reject H0).")
    else:
        print("\nSample too large for Shapiro-Wilk test. Consider Q-Q plot and other methods for normality assessment.")

    # Check dispersion pattern
    # For a count-like variable (0 to 9), compare mean and variance:
    if abs(var_val - mean_val) < mean_val * 0.1:
        print("\nVariance ~ mean. Poisson might be a starting guess.")
    elif var_val > mean_val:
        print("\nVariance > mean. Overdispersion: Negative Binomial or Quasi-Poisson could be considered.")
    else:
        print("\nVariance < mean. Underdispersion may need special handling (e.g. binomial or other model).")

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
    check_variability(df, measure_col)

    # Evaluate the distribution of the representative measure
    evaluate_distribution(df, measure_col)

    # Attempt fitting a model as before
    print("\nAttempting to fit a simple LMM with the representative measure...")
    try_fit_model(df, measure_col)

    print("\nIf the model is singular, consider:")
    print("- Removing or merging categories with zero or near-zero variance.")
    print("- Simplifying the model (e.g., remove random effects).")
    print("- Checking if certain outlets always have identical scores.")
    print("- Adjusting the threshold or removing problematic outlets/categories.")

if __name__ == "__main__":
    main()
