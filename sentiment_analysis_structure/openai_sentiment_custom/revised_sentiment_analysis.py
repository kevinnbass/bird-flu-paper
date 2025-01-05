import json
import openai
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm
from openai.error import OpenAIError
import logging

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Configure logging
logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify that the API key was loaded successfully
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Input and output file paths
INPUT_FILE = 'processed_all_articles_with_quotes_sentiment_analysis.jsonl'  # Your current JSONL file
OUTPUT_FILE = 'processed_all_articles_with_fulltext_sentiment_analysis.jsonl'  # New output file in JSON Lines format

# OpenAI API parameters
MODEL = 'gpt-4o'  # Ensure this is the correct model name
MAX_RETRIES = 1    # Number of retries for API calls
SLEEP_TIME = 5     # Wait time before retrying (in seconds)

# Cost parameters
INPUT_TOKEN_COST_PER_M = 1.25    # $1.25 per 1M input tokens
OUTPUT_TOKEN_COST_PER_M = 5.00   # $5.00 per 1M output tokens

# Initialize total tokens counters
total_input_tokens = 0
total_output_tokens = 0

# ------------------------------ #
#          Prompt Definition      #
# ------------------------------ #

# Prompt for Emotion and Sentiment Analysis of Fulltext
EMOTION_SENTIMENT_PROMPT = """
Assess the text's overall emotional tenor based on the following eight emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation. Additionally, evaluate the negative and positive sentiments of the text.

For each emotion and sentiment, provide a ranking on a scale from 0 to 9, where:
- 0 indicates absence
- 9 indicates an extremely strong positive presence.

**Instructions:**
- No Markdown, code fences, or extra text.
- Respond only with a JSON object containing:
  - "joy": integer between 0 and 9
  - "sadness": integer between 0 and 9
  - "anger": integer between 0 and 9
  - "fear": integer between 0 and 9
  - "surprise": integer between 0 and 9
  - "disgust": integer between 0 and 9
  - "trust": integer between 0 and 9
  - "anticipation": integer between 0 and 9
  - "negative_sentiment": integer between 0 and 9
  - "positive_sentiment": integer between 0 and 9

**Fulltext:**
{fulltext}
"""

# ------------------------------ #
#          Helper Function        #
# ------------------------------ #

def make_api_call(prompt_template, fulltext):
    """
    Makes an API call with the given prompt and returns the response.

    Args:
        prompt_template (str): The prompt template with placeholders.
        fulltext (str): The fulltext of the article.

    Returns:
        dict: Parsed JSON response from the API.
        int: Number of input tokens used.
        int: Number of output tokens generated.
    """
    prompt = prompt_template.format(fulltext=fulltext)

    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an analytical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,  # Increased to accommodate more detailed responses
            )
            reply = response['choices'][0]['message']['content'].strip()

            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

            global total_input_tokens, total_output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"Raw API Response for Fulltext:\n{reply}\n")
            logging.info(f"API Response for Fulltext:\n{reply}\n")
            logging.info(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")

            # Clean reply if it's enclosed in code fences
            if reply.startswith("```") and reply.endswith("```"):
                reply = '\n'.join(reply.split('\n')[1:-1]).strip()

            if not (reply.startswith('{') and reply.endswith('}')):
                raise ValueError("Response is not a valid JSON object.")

            result = json.loads(reply)

            # Validate that all required fields are present and within the specified range
            required_fields = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "negative_sentiment", "positive_sentiment"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field '{field}' in response.")
                if not isinstance(result[field], int) or not 0 <= result[field] <= 9:
                    raise ValueError(f"Field '{field}' must be an integer between 0 and 9.")

            return result, input_tokens, output_tokens

        except (OpenAIError, ValueError, json.JSONDecodeError) as e:
            logging.error(f"Error during API call: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Error: {e}. Retrying in {SLEEP_TIME} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})\n")
                time.sleep(SLEEP_TIME)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Returning empty result for this fulltext.\n")
                logging.error(f"Failed after {MAX_RETRIES} attempts for fulltext.")
                return {}, 0, 0

def analyze_fulltext(fulltext):
    """
    Analyzes the fulltext using the emotion and sentiment prompt.

    Args:
        fulltext (str): The fulltext of the article.

    Returns:
        dict: Results containing emotion and sentiment ratings.
        int: Input tokens used.
        int: Output tokens generated.
    """
    result, input_tokens, output_tokens = make_api_call(EMOTION_SENTIMENT_PROMPT, fulltext)
    return result, input_tokens, output_tokens

# ------------------------------ #
#           Main Execution        #
# ------------------------------ #

def main():
    global total_input_tokens, total_output_tokens

    # Ensure output file exists and is ready for appending
    if not os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            pass  # Create an empty JSONL file
        logging.info(f"Created new output file: {OUTPUT_FILE}")

    # Open the output file in append mode
    output_handle = open(OUTPUT_FILE, 'a', encoding='utf-8')

    # Load the input JSONL data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            articles = infile.readlines()  # Read all lines
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        logging.error(f"Input file '{INPUT_FILE}' not found.")
        output_handle.close()
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        logging.error(f"Error reading input file: {e}")
        output_handle.close()
        return

    num_articles = len(articles)
    print(f"Total articles loaded: {num_articles}\n")
    logging.info(f"Total articles loaded: {num_articles}\n")

    if num_articles == 0:
        print("No articles found in the input file. Exiting.")
        logging.info("No articles found in the input file. Exiting.")
        output_handle.close()
        return

    # Process each article with a progress bar
    for idx, line in enumerate(tqdm(articles, desc="Processing Articles")):
        try:
            article = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {idx + 1}: {e}. Skipping this article.\n")
            logging.error(f"Error decoding JSON on line {idx + 1}: {e}. Skipping this article.\n")
            continue

        fulltext = article.get('fulltext', '').strip()

        if not fulltext:
            print(f"Article {idx + 1} has no fulltext. Assigning default values.\n")
            logging.info(f"Article {idx + 1} has no fulltext. Assigning default values.\n")
            # Assign default scores
            emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "negative_sentiment", "positive_sentiment"]
            for emotion in emotions:
                score_key = f"{emotion}_fulltext"
                article[score_key] = None  # Assigning null

            # Add feedback
            article['feedback_sa_fulltext'] = 'No fulltext provided.'

        else:
            # Analyze the fulltext using the prompt
            combined_result, input_tokens, output_tokens = analyze_fulltext(fulltext)

            if combined_result:
                # Update the article with new fields for fulltext
                for emotion, score in combined_result.items():
                    score_key = f"{emotion}_fulltext"
                    article[score_key] = score

                # Add feedback
                article['feedback_sa_fulltext'] = 'Emotion and sentiment analysis completed successfully.'
            else:
                # Assign default values in case of failure
                emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "negative_sentiment", "positive_sentiment"]
                for emotion in emotions:
                    score_key = f"{emotion}_fulltext"
                    article[score_key] = None  # Assigning null

                # Add feedback
                article['feedback_sa_fulltext'] = 'Failed to analyze emotion and sentiment.'

            # Log the results
            if combined_result:
                logging.info(f"Article {idx + 1} Fulltext Results:\n" +
                             ", ".join([f"{emotion}_fulltext: {score}" for emotion, score in combined_result.items()]) +
                             f"\nFeedback_sa_fulltext: {article.get('feedback_sa_fulltext', '')}\n" +
                             f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
                print(f"Article {idx + 1} - Emotion and Sentiment Ratings Assigned.\n"
                      f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
            else:
                logging.info(f"Article {idx + 1} Fulltext Analysis Failed.\n" +
                             f"Feedback_sa_fulltext: {article.get('feedback_sa_fulltext', '')}\n" +
                             f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
                print(f"Article {idx + 1} - Failed to assign Emotion and Sentiment Ratings.\n"
                      f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")

        # Write the updated article to output immediately
        output_handle.write(json.dumps(article) + '\n')

    # Close the output file
    output_handle.close()

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_M
    total_cost = input_cost + output_cost

    # Print summary
    print("\n--- Processing Summary ---")
    print(f"Total articles processed: {num_articles}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Cost: ${total_cost:.6f}")
    print("--------------------------\n")

    # Log summary
    logging.info("--- Processing Summary ---")
    logging.info(f"Total articles processed: {num_articles}")
    logging.info(f"Total Input Tokens: {total_input_tokens}")
    logging.info(f"Total Output Tokens: {total_output_tokens}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    logging.info("--------------------------\n")

if __name__ == "__main__":
    main()
