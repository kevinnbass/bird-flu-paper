# sentiment_analysis_module.py

import logging
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
from transformers import pipeline
import torch
from utils import logger
import plotly.express as px
import time
import hashlib
import sys  # Added for handling non-interactive scenarios

def compute_file_hash(file_path, hash_algo='sha256'):
    """
    Compute the hash of a file using the specified hashing algorithm.

    Parameters:
        file_path (str): Path to the file.
        hash_algo (str): Hashing algorithm to use (default: 'sha256').

    Returns:
        str: Hexadecimal hash string of the file content.
    """
    hash_func = hashlib.new(hash_algo)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def prompt_user(prompt_message, default='n'):
    """
    Prompt the user with a yes/no question and return the response.

    Parameters:
        prompt_message (str): The message to display to the user.
        default (str): The default answer if the user just presses Enter.

    Returns:
        bool: True if the user responds with 'y' or 'Y', False otherwise.
    """
    valid = {"y": True, "n": False}
    prompt = f"{prompt_message} [{'Y' if default == 'y' else 'y'}/{'N' if default == 'n' else 'n'}]: "
    while True:
        choice = input(prompt).strip().lower()
        if not choice:
            choice = default
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'y' or 'n'.")

def apply_sentiment_analysis(df, cache_path="cache/sentiment_analysis.parquet",
                             source_path="all_articles_bird_flu_project_for_sa.json",
                             hash_path="cache/sentiment_analysis.hash",
                             max_age_hours=24,
                             force_regen=False):
    """
    Apply multiple sentiment analysis methods to the DataFrame per media category.
    Cache the results to avoid redundant computations.
    Save interactive Plotly plots and log sentiment distributions.

    Parameters:
        df (pd.DataFrame): DataFrame containing the articles.
        cache_path (str): Path to the cache file.
        source_path (str): Path to the source data file to check for updates.
        hash_path (str): Path to store the hash of the source data.
        max_age_hours (int): Maximum age of cache in hours before regeneration.
        force_regen (bool): If True, force regeneration of the cache without prompting.

    Returns:
        pd.DataFrame: DataFrame with additional sentiment analysis columns per category.
    """
    try:
        # **Move current_data_hash computation to the beginning**
        current_data_hash = compute_file_hash(source_path)
        logger.debug(f"Computed current data hash: {current_data_hash}")

        regenerate_cache = force_regen

        if not force_regen:
            # Existing cache validation and prompt logic...
            # Detailed Cache Validation Logging
            logger.info("Starting cache validation process...")

            # Check if cache and hash files exist
            if os.path.exists(cache_path) and os.path.exists(hash_path):
                logger.info(f"Cache file found at '{cache_path}'.")
                logger.info(f"Hash file found at '{hash_path}'.")

                # Load stored hash
                with open(hash_path, 'r') as hash_file:
                    stored_data_hash = hash_file.read().strip()
                logger.debug(f"Stored data hash: {stored_data_hash}")

                # Compare hashes
                if current_data_hash == stored_data_hash:
                    logger.info("Data content has not changed since the last cache. Loading from cache.")
                    try:
                        df_cached = pd.read_parquet(cache_path)
                        logger.info(f"Loaded sentiment analysis results from cache with {len(df_cached)} records.")

                        # Drop unwanted 'Unnamed:' columns in cached data
                        unnamed_cols_cached = [col for col in df_cached.columns if col.startswith('Unnamed:')]
                        if unnamed_cols_cached:
                            df_cached = df_cached.drop(columns=unnamed_cols_cached)
                            logger.info(f"Dropped unwanted columns from cached data: {', '.join(unnamed_cols_cached)}")
                        else:
                            logger.info("No 'Unnamed:' columns found in cached data to drop.")

                        # Ensure 'word_count' is numeric in cached data
                        if 'word_count' in df_cached.columns:
                            df_cached['word_count'] = pd.to_numeric(df_cached['word_count'], errors='coerce').fillna(0).astype(int)
                            logger.info("'word_count' column in cached data converted to numeric type successfully.")
                        else:
                            logger.warning("'word_count' column not found in cached data.")

                        return df_cached
                    except Exception as e:
                        logger.error(f"Failed to load from cache: {e}. Proceeding to regenerate cache.")
                        regenerate_cache = True
                else:
                    logger.info("Data content has changed since the last cache. Regenerating cache.")
                    regenerate_cache = True
            else:
                if not os.path.exists(cache_path):
                    logger.info(f"No cache file found at '{cache_path}'. Proceeding to perform sentiment analysis.")
                if not os.path.exists(hash_path):
                    logger.info(f"No hash file found at '{hash_path}'. Proceeding to perform sentiment analysis.")
                regenerate_cache = True

        if regenerate_cache:
            if not force_regen:
                # Prompt the user for manual override
                try:
                    user_choice = prompt_user(
                        "Cache is outdated or data has been updated. Do you want to regenerate the cache?",
                        default='y'
                    )
                except KeyboardInterrupt:
                    logger.error("User aborted the process.")
                    print("\nProcess aborted by user.")
                    sys.exit(1)

                if user_choice:
                    logger.info("User chose to regenerate the cache.")
                    # Proceed to regenerate the cache
                else:
                    logger.info("User chose not to regenerate the cache. Attempting to load existing cache.")
                    if os.path.exists(cache_path):
                        try:
                            df_cached = pd.read_parquet(cache_path)
                            logger.info(f"Loaded sentiment analysis results from cache with {len(df_cached)} records.")

                            # Drop unwanted 'Unnamed:' columns in cached data
                            unnamed_cols_cached = [col for col in df_cached.columns if col.startswith('Unnamed:')]
                            if unnamed_cols_cached:
                                df_cached = df_cached.drop(columns=unnamed_cols_cached)
                                logger.info(f"Dropped unwanted columns from cached data: {', '.join(unnamed_cols_cached)}")
                            else:
                                logger.info("No 'Unnamed:' columns found in cached data to drop.")

                            # Ensure 'word_count' is numeric in cached data
                            if 'word_count' in df_cached.columns:
                                df_cached['word_count'] = pd.to_numeric(df_cached['word_count'], errors='coerce').fillna(0).astype(int)
                                logger.info("'word_count' column in cached data converted to numeric type successfully.")
                            else:
                                logger.warning("'word_count' column not found in cached data.")

                            return df_cached
                        except Exception as e:
                            logger.error(f"Failed to load from cache: {e}. Regenerating cache.")
                            user_force = prompt_user(
                                "Failed to load from cache. Do you want to regenerate the cache?",
                                default='y'
                            )
                            if user_force:
                                logger.info("User chose to regenerate the cache after cache loading failure.")
                                regenerate_cache = True
                            else:
                                logger.error("Cannot proceed without a valid cache. Exiting.")
                                print("Cannot proceed without a valid cache. Exiting.")
                                sys.exit(1)
                    else:
                        logger.error("Cache file does not exist. Cannot load cached data.")
                        user_regen = prompt_user(
                            "Cache file does not exist. Do you want to perform sentiment analysis and create a new cache?",
                            default='y'
                        )
                        if user_regen:
                            logger.info("User chose to regenerate the cache.")
                            regenerate_cache = True
                        else:
                            logger.error("Cannot proceed without performing sentiment analysis. Exiting.")
                            print("Cannot proceed without performing sentiment analysis. Exiting.")
                            sys.exit(1)

            if regenerate_cache:
                logger.info("Starting sentiment analysis and cache regeneration process...")

                # Initialize sentiment analyzers
                vader_analyzer = SentimentIntensityAnalyzer()
                afinn_analyzer = Afinn()

                # Initialize Transformer-based sentiment analyzer
                try:
                    transformer_pipeline = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=0 if torch.cuda.is_available() else -1,
                    )
                    logger.info("Transformer-based sentiment analyzer initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize Transformer pipeline: {e}")
                    transformer_pipeline = None
                    logger.warning("Transformer-based sentiment analysis will be skipped.")

                # Define media outlet categories
                categories = df['media_category'].unique()
                logger.info(f"Found {len(categories)} unique media categories for sentiment analysis.")

                # Number of parallel jobs
                num_jobs = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
                logger.info(f"Using {num_jobs} parallel jobs for sentiment analysis.")

                for category in categories:
                    logger.info(f"Processing sentiment analysis for category: {category}")
                    subset = df[df['media_category'] == category]
                    if subset.empty:
                        logger.warning(f"No data found for category: {category}. Skipping sentiment analysis.")
                        continue

                    # VADER Sentiment on Titles
                    logger.info(f"Applying VADER Sentiment Analysis on Titles for category: {category}...")
                    subset_vader_title = Parallel(n_jobs=num_jobs)(
                        delayed(get_vader_sentiment)(text, vader_analyzer)
                        for text in tqdm(subset["clean_title"], desc=f"VADER Titles - {category}")
                    )
                    df.loc[subset.index, f"vader_compound_title_{category}"] = [
                        sentiment["compound"] for sentiment in subset_vader_title
                    ]

                    # VADER Sentiment on Fulltexts
                    logger.info(f"Applying VADER Sentiment Analysis on Fulltexts for category: {category}...")
                    subset_vader_fulltext = Parallel(n_jobs=num_jobs)(
                        delayed(get_vader_sentiment)(text, vader_analyzer)
                        for text in tqdm(subset["clean_fulltext"], desc=f"VADER Fulltexts - {category}")
                    )
                    df.loc[subset.index, f"vader_compound_fulltext_{category}"] = [
                        sentiment["compound"] for sentiment in subset_vader_fulltext
                    ]

                    # TextBlob Sentiment on Titles
                    logger.info(f"Applying TextBlob Sentiment Analysis on Titles for category: {category}...")
                    subset_textblob_title = Parallel(n_jobs=num_jobs)(
                        delayed(get_textblob_sentiment)(text)
                        for text in tqdm(subset["clean_title"], desc=f"TextBlob Titles - {category}")
                    )
                    df.loc[subset.index, f"textblob_polarity_title_{category}"] = [
                        sentiment.polarity for sentiment in subset_textblob_title
                    ]
                    df.loc[subset.index, f"textblob_subjectivity_title_{category}"] = [
                        sentiment.subjectivity for sentiment in subset_textblob_title
                    ]

                    # TextBlob Sentiment on Fulltexts
                    logger.info(f"Applying TextBlob Sentiment Analysis on Fulltexts for category: {category}...")
                    subset_textblob_fulltext = Parallel(n_jobs=num_jobs)(
                        delayed(get_textblob_sentiment)(text)
                        for text in tqdm(subset["clean_fulltext"], desc=f"TextBlob Fulltexts - {category}")
                    )
                    df.loc[subset.index, f"textblob_polarity_fulltext_{category}"] = [
                        sentiment.polarity for sentiment in subset_textblob_fulltext
                    ]
                    df.loc[subset.index, f"textblob_subjectivity_fulltext_{category}"] = [
                        sentiment.subjectivity for sentiment in subset_textblob_fulltext
                    ]

                    # AFINN Sentiment on Titles
                    logger.info(f"Applying AFINN Sentiment Analysis on Titles for category: {category}...")
                    subset_afinn_title = Parallel(n_jobs=num_jobs)(
                        delayed(get_afinn_sentiment)(text, afinn_analyzer)
                        for text in tqdm(subset["clean_title"], desc=f"AFINN Titles - {category}")
                    )
                    df.loc[subset.index, f"afinn_sentiment_title_{category}"] = subset_afinn_title

                    # AFINN Sentiment on Fulltexts
                    logger.info(f"Applying AFINN Sentiment Analysis on Fulltexts for category: {category}...")
                    subset_afinn_fulltext = Parallel(n_jobs=num_jobs)(
                        delayed(get_afinn_sentiment)(text, afinn_analyzer)
                        for text in tqdm(subset["clean_fulltext"], desc=f"AFINN Fulltexts - {category}")
                    )
                    df.loc[subset.index, f"afinn_sentiment_fulltext_{category}"] = subset_afinn_fulltext

                    # Transformer-Based Sentiment Analysis on Titles
                    if transformer_pipeline:
                        logger.info(f"Applying Transformer-Based Sentiment Analysis on Titles for category: {category}...")
                        subset_transformer_title = Parallel(n_jobs=num_jobs)(
                            delayed(sentiment_pipeline_transformer)(text[:512], transformer_pipeline)
                            for text in tqdm(subset["clean_title"], desc=f"Transformer Titles - {category}")
                        )
                        df.loc[subset.index, f"transformer_label_title_{category}"] = [
                            sentiment["label"] for sentiment in subset_transformer_title
                        ]
                        df.loc[subset.index, f"transformer_score_title_{category}"] = [
                            sentiment["score"] for sentiment in subset_transformer_title
                        ]

                        # Transformer-Based Sentiment Analysis on Fulltexts
                        logger.info(f"Applying Transformer-Based Sentiment Analysis on Fulltexts for category: {category}...")
                        subset_transformer_fulltext = Parallel(n_jobs=num_jobs)(
                            delayed(sentiment_pipeline_transformer)(text[:512], transformer_pipeline)
                            for text in tqdm(subset["clean_fulltext"], desc=f"Transformer Fulltexts - {category}")
                        )
                        df.loc[subset.index, f"transformer_label_fulltext_{category}"] = [
                            sentiment["label"] for sentiment in subset_transformer_fulltext
                        ]
                        df.loc[subset.index, f"transformer_score_fulltext_{category}"] = [
                            sentiment["score"] for sentiment in subset_transformer_fulltext
                        ]
                    else:
                        logger.warning("Transformer pipeline not initialized. Skipping Transformer-based sentiment analysis.")

                    # Generate and Save Sentiment Distribution Plots
                    generate_sentiment_distribution_plots(df, category)

                logger.info("Sentiment analysis methods applied successfully per category.")

                # Ensure 'word_count' is numeric before saving to cache
                if 'word_count' in df.columns:
                    df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0).astype(int)
                    logger.info("'word_count' column converted to numeric type successfully before caching.")
                else:
                    logger.warning("'word_count' column not found in DataFrame before caching.")

                # Drop any remaining unwanted 'Unnamed:' columns before caching
                unnamed_cols_final = [col for col in df.columns if col.startswith('Unnamed:')]
                if unnamed_cols_final:
                    df = df.drop(columns=unnamed_cols_final)
                    logger.info(f"Dropped unwanted columns before caching: {', '.join(unnamed_cols_final)}")
                else:
                    logger.info("No 'Unnamed:' columns found in DataFrame before caching.")

                # Save the processed DataFrame to cache
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_parquet(cache_path, compression='snappy')
                logger.info(f"Sentiment analysis results cached to '{cache_path}'.")

                # Save the current data hash to the hash file
                with open(hash_path, 'w') as hash_file:
                    hash_file.write(current_data_hash)
                logger.info(f"Data hash '{current_data_hash}' saved to '{hash_path}'.")

                return df

    except Exception as e:
        logger.error(f"Error during sentiment analysis application: {e}")
        raise

def get_vader_sentiment(text, analyzer):
    """
    Get VADER sentiment scores for a given text.

    Parameters:
        text (str): The text to analyze.
        analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer.

    Returns:
        dict: VADER sentiment scores.
    """
    try:
        return analyzer.polarity_scores(text)
    except Exception as e:
        logger.warning(f"VADER sentiment analysis failed for text: {text[:30]}... Error: {e}")
        return {"compound": 0.0, "neg": 0.0, "neu": 0.0, "pos": 0.0}

def get_textblob_sentiment(text):
    """
    Get TextBlob sentiment polarity and subjectivity for a given text.

    Parameters:
        text (str): The text to analyze.

    Returns:
        TextBlob: Sentiment object containing polarity and subjectivity.
    """
    try:
        return TextBlob(text).sentiment
    except Exception as e:
        logger.warning(f"TextBlob sentiment analysis failed for text: {text[:30]}... Error: {e}")
        return TextBlob("Neutral").sentiment  # Return neutral sentiment

def get_afinn_sentiment(text, afinn_analyzer):
    """
    Get AFINN sentiment score for a given text.

    Parameters:
        text (str): The text to analyze.
        afinn_analyzer (Afinn): AFINN sentiment analyzer.

    Returns:
        float: AFINN sentiment score.
    """
    try:
        return afinn_analyzer.score(text)
    except Exception as e:
        logger.warning(f"AFINN sentiment analysis failed for text: {text[:30]}... Error: {e}")
        return 0.0  # Neutral score

def sentiment_pipeline_transformer(text, pipeline_instance):
    """
    Get Transformer-based sentiment label and score for a given text.

    Parameters:
        text (str): The text to analyze.
        pipeline_instance (pipeline): Transformer sentiment analysis pipeline.

    Returns:
        dict: Sentiment label and score.
    """
    try:
        sentiment = pipeline_instance(text)[0]
        return {
            "label": sentiment["label"],
            "score": sentiment["score"]
        }
    except Exception as e:
        logger.warning(f"Transformer sentiment analysis failed for text: {text[:30]}... Error: {e}")
        return {"label": "NEUTRAL", "score": 0.0}  # Default neutral sentiment

def generate_sentiment_distribution_plots(df, category):
    """
    Generate and save Plotly histograms for sentiment scores.

    Parameters:
        df (pd.DataFrame): DataFrame containing sentiment analysis results.
        category (str): The media category.

    Returns:
        None
    """
    try:
        logger.info(f"Generating sentiment distribution plots for category: {category}...")

        # Define sentiment methods and their corresponding metrics
        sentiment_methods = {
            'vader': ['compound'],
            'textblob': ['polarity', 'subjectivity'],
            'afinn': ['sentiment'],
            'transformer': ['score']
        }

        for method, metrics in sentiment_methods.items():
            for metric in metrics:
                # Define column name based on method, metric, and category
                if method == 'transformer' and metric == 'score':
                    column_name = f"{method}_score_fulltext_{category}"
                elif method == 'transformer' and metric == 'label':
                    column_name = f"{method}_label_fulltext_{category}"
                elif method == 'afinn' and metric == 'sentiment':
                    column_name = f"{method}_sentiment_fulltext_{category}"
                else:
                    column_name = f"{method}_{metric}_fulltext_{category}"

                if column_name in df.columns:
                    fig = px.histogram(
                        df,
                        x=column_name,
                        nbins=50,
                        title=f"{method.capitalize()} {metric.capitalize()} Distribution for {category}",
                        labels={column_name: f"{method.capitalize()} {metric.capitalize()}"}
                    )
                    plot_path = os.path.join("plots", "sentiment_analysis", f"{method}_{metric}_distribution_{category}.html")
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    fig.write_html(plot_path)
                    logger.info(f"Sentiment distribution plot saved to '{plot_path}' for {method} {metric} in {category}.")
                else:
                    logger.warning(f"Column '{column_name}' not found in DataFrame. Skipping plot generation for this metric.")
    except Exception as e:
        logger.error(f"Error generating sentiment distribution plots for category {category}: {e}")
        raise
