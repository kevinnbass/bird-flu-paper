import logging
import sys
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler for logging
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------------------------------------------------------------
# Load and Prepare Data
# ---------------------------------------------------------------------
# Replace this with your actual data loading steps. For example:
# df = pd.read_json('articles.jsonl', lines=True)

# For demonstration, let's assume `df` is already a pandas DataFrame
# with the following columns:
# 'media_outlet': categorical variable for media sources
# 'response': numeric response variable
# 'predictor1', 'predictor2', etc.: predictors
# Ensure your dataframe `df` is defined here.

# Example placeholder:
# df = pd.read_csv('your_prepared_data.csv')

# Check that required columns exist:
required_columns = ['media_outlet', 'response', 'predictor1']
for col in required_columns:
    if col not in df.columns:
        logger.error(f"Column '{col}' not found in dataframe. Please ensure your data is prepared correctly.")
        sys.exit(1)

logger.info(f"Total articles loaded: {len(df)}")

# ---------------------------------------------------------------------
# Map media outlets to categories if needed
# (Assume your mapping code is here, if required)
# logger.info("Media outlets mapped to categories successfully.")

# ---------------------------------------------------------------------
# Filter Outlets with < 2 Observations
# ---------------------------------------------------------------------
outlet_counts = df['media_outlet'].value_counts()
outlets_to_remove = outlet_counts[outlet_counts < 2].index.tolist()

if outlets_to_remove:
    logger.info("Filtering out outlets with fewer than 2 observations:")
    for outlet in outlets_to_remove:
        obs_count = outlet_counts[outlet]
        logger.info(f" - Removing {outlet}: {obs_count} observation(s)")
    # Filter the DataFrame
    initial_count = len(df)
    df = df[~df['media_outlet'].isin(outlets_to_remove)]
    final_count = len(df)
    removed_count = initial_count - final_count
    logger.info(f"Filtered out {removed_count} total observations from {len(outlets_to_remove)} outlet(s).")
else:
    logger.info("No outlets with fewer than 2 observations found.")

# ---------------------------------------------------------------------
# Proceed with Mean/Median Calculations or Other Stats (Optional)
# (Assume code for aggregations, plotting, etc., if needed)
# ---------------------------------------------------------------------
# Example:
# aggregated = df.groupby('media_outlet')['response'].agg(['mean', 'median'])
# logger.info("Mean and median statistics calculated successfully.")

# ---------------------------------------------------------------------
# Fit the Linear Mixed Model
# ---------------------------------------------------------------------
# Example formula: 'response ~ predictor1 + (1|media_outlet)'
# In statsmodels, random effects are specified differently:
# You use MixedLM(endog, exog, groups=...).
# If using formula interface:
# smf.mixedlm("response ~ predictor1", df, groups=df["media_outlet"]).fit()

try:
    logger.info("Fitting linear mixed model...")
    model = smf.mixedlm("response ~ predictor1", data=df, groups=df["media_outlet"])
    result = model.fit(reml=False)  # or True depending on your scenario
    logger.info("LMM fitted successfully.")
    logger.info(result.summary())
except Exception as e:
    logger.error(f"Error fitting LMM: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# If warnings about singular matrices remain, consider simplifying the model,
# checking collinearity, or further filtering.
# ---------------------------------------------------------------------

logger.info("Analysis completed successfully.")
