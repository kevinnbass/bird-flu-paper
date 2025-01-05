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
INPUT_FILE = 'processed_all_articles_fixed_4.jsonl'   # Your current JSONL file
OUTPUT_FILE = 'processed_all_articles_fixed_5.jsonl'        # New output file

# OpenAI API parameters
MODEL = 'gpt-4o'  # Ensure this is the correct model name, or use an available one like "gpt-3.5-turbo"
MAX_RETRIES = 1   # Number of retries for API calls
SLEEP_TIME = 5    # Wait time before retrying (in seconds)

# Cost parameters
INPUT_TOKEN_COST_PER_M = 1.25    # $1.25 per 1M input tokens
OUTPUT_TOKEN_COST_PER_M = 5.00   # $5.00 per 1M output tokens

# Initialize total tokens counters
total_input_tokens = 0
total_output_tokens = 0

# ------------------------------ #
#          Prompt Definition     #
# ------------------------------ #

# Prompt for Emotion and Sentiment Analysis of the Title
EMOTION_SENTIMENT_PROMPT_TITLE = """
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

**Title:**
{title}
"""

# ------------------------------ #
#          Helper Functions      #
# ------------------------------ #

def make_api_call(prompt_template, text):
    """
    Makes an API call with the given prompt and returns the response.

    Args:
        prompt_template (str): The prompt template with placeholders.
        text (str): The text (in this case, the title) to be analyzed.

    Returns:
        dict: Parsed JSON response from the API.
        int: Number of input tokens used.
        int: Number of output tokens generated.
    """
    prompt = prompt_template.format(title=text)

    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an analytical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,  # Enough to handle the JSON response
            )
            reply = response['choices'][0]['message']['content'].strip()

            usage = response.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

            global total_input_tokens, total_output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"Raw API Response for Title:\n{reply}\n")
            logging.info(f"API Response for Title:\n{reply}\n")
            logging.info(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")

            # Clean reply if it's enclosed in code fences
            if reply.startswith("```") and reply.endswith("```"):
                reply = '\n'.join(reply.split('\n')[1:-1]).strip()

            # Basic validation check
            if not (reply.startswith('{') and reply.endswith('}')):
                raise ValueError("Response is not a valid JSON object.")

            result = json.loads(reply)

            # Validate that all required fields are present and within the specified range
            required_fields = [
                "joy", "sadness", "anger", "fear", "surprise",
                "disgust", "trust", "anticipation",
                "negative_sentiment", "positive_sentiment"
            ]
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
                print(f"Failed after {MAX_RETRIES} attempts. Returning empty result for this title.\n")
                logging.error(f"Failed after {MAX_RETRIES} attempts for this title.")
                return {}, 0, 0

def analyze_title(title):
    """
    Analyzes the title text using the emotion and sentiment prompt.

    Args:
        title (str): The title of the article.

    Returns:
        dict: Results containing emotion and sentiment ratings.
        int: Input tokens used.
        int: Output tokens generated.
    """
    result, input_tokens, output_tokens = make_api_call(EMOTION_SENTIMENT_PROMPT_TITLE, title)
    return result, input_tokens, output_tokens

# ------------------------------ #
#           Main Execution       #
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

    # Define which fields we expect for Title-based analysis
    required_title_fields = [
        "joy_title", "sadness_title", "anger_title", "fear_title", "surprise_title",
        "disgust_title", "trust_title", "anticipation_title",
        "negative_sentiment_title", "positive_sentiment_title"
    ]

    # Process each article with a progress bar
    for idx, line in enumerate(tqdm(articles, desc="Processing Articles")):
        try:
            article = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {idx + 1}: {e}. Skipping this article.\n")
            logging.error(f"Error decoding JSON on line {idx + 1}: {e}. Skipping this article.\n")
            continue

        # Check if this article already has the required title sentiment fields
        if all(field in article for field in required_title_fields):
            # Already processed, so just write it to the output
            output_handle.write(json.dumps(article) + '\n')
            continue

        # Get the title
        title_text = article.get('title', '').strip()

        # If there's no title, assign null to each new field
        if not title_text:
            print(f"Article {idx + 1} has no title. Assigning default values.\n")
            logging.info(f"Article {idx + 1} has no title. Assigning default values.\n")

            for field in required_title_fields:
                article[field] = None

            # Add feedback
            article['feedback_sa_title'] = 'No title provided.'
            output_handle.write(json.dumps(article) + '\n')
            continue

        # Otherwise, analyze the title
        combined_result, input_tokens, output_tokens = analyze_title(title_text)

        if combined_result:
            # Update the article with new fields for the title
            for emotion, score in combined_result.items():
                score_key = f"{emotion}_title"
                article[score_key] = score

            # Add feedback
            article['feedback_sa_title'] = 'Emotion and sentiment analysis of title completed successfully.'

            # Log the results
            logging.info(
                f"Article {idx + 1} Title Results:\n" +
                ", ".join([f"{k}_title: {v}" for k, v in combined_result.items()]) +
                f"\nfeedback_sa_title: {article.get('feedback_sa_title', '')}\n" +
                f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n"
            )
            print(
                f"Article {idx + 1} - Title Emotion/Sentiment Ratings Assigned.\n"
                f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n"
            )
        else:
            # Assign default values if the API call failed
            for field in required_title_fields:
                article[field] = None

            article['feedback_sa_title'] = 'Failed to analyze emotion and sentiment for title.'
            logging.info(
                f"Article {idx + 1} Title Analysis Failed.\n" +
                f"feedback_sa_title: {article.get('feedback_sa_title', '')}\n" +
                f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n"
            )
            print(
                f"Article {idx + 1} - Failed to assign Title Emotion/Sentiment Ratings.\n"
                f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n"
            )

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
