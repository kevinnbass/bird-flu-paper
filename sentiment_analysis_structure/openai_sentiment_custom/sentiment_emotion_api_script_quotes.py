import json
import openai
import time
import os
import re
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
INPUT_FILE = 'processed_all_articles_with_quotes.json'  # Your current JSON file (standard JSON array)
OUTPUT_FILE = 'processed_all_articles_with_quotes_sentiment_analysis.jsonl'  # Output file in JSON Lines format

# OpenAI API parameters
MODEL = 'gpt-4o'  # Corrected model name
MAX_RETRIES = 1   # Increased retries for better resilience
SLEEP_TIME = 5    # Fixed wait time before retrying (in seconds)

# Cost parameters
INPUT_TOKEN_COST_PER_M = 1.25   # $1.25 per 1M input tokens
OUTPUT_TOKEN_COST_PER_M = 5.00  # $5.00 per 1M output tokens

# Initialize total tokens counters
total_input_tokens = 0
total_output_tokens = 0

# ------------------------------ #
#          Prompt Definition      #
# ------------------------------ #

# New Prompt for Emotion and Sentiment Analysis per Quotation
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

**Quotation Text:**
{quotation_text}
"""

# ------------------------------ #
#          Helper Function        #
# ------------------------------ #

def make_api_call(prompt_template, quotation_text):
    """
    Makes an API call with the given prompt and returns the response.

    Args:
        prompt_template (str): The prompt template with placeholders.
        quotation_text (str): The text of the quotation.

    Returns:
        dict: Parsed JSON response from the API.
        int: Number of input tokens used.
        int: Number of output tokens generated.
    """
    prompt = prompt_template.format(quotation_text=quotation_text)

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

            print(f"Raw API Response for Quotation:\n{reply}\n")
            logging.info(f"API Response for Quotation:\n{reply}\n")
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
                print(f"Failed after {MAX_RETRIES} attempts. Returning empty result for this quotation.\n")
                logging.error(f"Failed after {MAX_RETRIES} attempts for quotation.")
                return {}, 0, 0

def analyze_quotation(quotation_text):
    """
    Analyzes the quotation using the emotion and sentiment prompt.

    Args:
        quotation_text (str): The text of the quotation.

    Returns:
        dict: Results containing emotion and sentiment ratings.
        int: Input tokens used.
        int: Output tokens generated.
    """
    result, input_tokens, output_tokens = make_api_call(EMOTION_SENTIMENT_PROMPT, quotation_text)
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

    # Load the input JSON data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            articles = json.load(infile)  # Load the entire JSON array
    except FileNotFoundError:
        print(f"Input file '{INPUT_FILE}' not found.")
        logging.error(f"Input file '{INPUT_FILE}' not found.")
        output_handle.close()
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        logging.error(f"Error decoding JSON: {e}")
        output_handle.close()
        return

    # Separate articles into already processed and to be processed
    already_processed = [article for article in articles if any(re.match(r'joy_\d+', key) for key in article.keys())]
    to_process_articles = [article for article in articles if not any(re.match(r'joy_\d+', key) for key in article.keys())]

    num_already_processed = len(already_processed)
    num_to_process = len(to_process_articles)

    print(f"Total articles loaded: {len(articles)}")
    print(f"Articles already processed (with emotion and sentiment ratings): {num_already_processed}")
    print(f"Articles to be processed (without emotion and sentiment ratings): {num_to_process}\n")
    logging.info(f"Total articles loaded: {len(articles)}")
    logging.info(f"Articles already processed (with emotion and sentiment ratings): {num_already_processed}")
    logging.info(f"Articles to be processed (without emotion and sentiment ratings): {num_to_process}\n")

    if num_to_process == 0:
        print("No unprocessed articles found. Exiting.")
        logging.info("No unprocessed articles found. Exiting.")
        output_handle.close()
        return

    # Regular expression to match quotation fields
    quotation_pattern = re.compile(r'quotation_(\d+)')

    # Process each unprocessed article with a progress bar
    for idx, article in enumerate(tqdm(to_process_articles, desc="Processing Unprocessed Articles")):
        # Find all quotation fields in the article
        quotation_fields = [key for key in article.keys() if quotation_pattern.match(key)]
        if not quotation_fields:
            print(f"Article {idx + 1} has no quotations. Assigning default values.\n")
            logging.info(f"Article {idx + 1} has no quotations. Assigning default values.\n")
            # Optionally, add feedback
            article['feedback_sa'] = 'No quotations provided.'
            # Write the updated article to output
            output_handle.write(json.dumps(article) + '\n')
            continue

        # Extract quotation numbers and sort them
        quotation_numbers = sorted([int(quotation_pattern.match(key).group(1)) for key in quotation_fields])
        max_quotation_num = max(quotation_numbers)

        # Process each quotation
        for q_num in quotation_numbers:
            quotation_key = f'quotation_{q_num}'
            quotation_text = article.get(quotation_key, '').strip()

            if not quotation_text:
                print(f"Article {idx + 1}, {quotation_key} is empty. Assigning default scores.\n")
                logging.info(f"Article {idx + 1}, {quotation_key} is empty. Assigning default scores.\n")
                # Assign default scores for this quotation
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "negative_sentiment", "positive_sentiment"]:
                    score_key = f"{emotion}_{q_num}"
                    article[score_key] = 0
                # **Changed feedback field name to feedback_sa_n**
                feedback_key = f'feedback_sa_{q_num}'
                article[feedback_key] = 'No quotation text provided.'
                continue

            # Analyze the quotation using the new prompt
            combined_result, input_tokens, output_tokens = analyze_quotation(quotation_text)

            if combined_result:
                # Update the article with new fields for this quotation
                for emotion, score in combined_result.items():
                    score_key = f"{emotion}_{q_num}"
                    article[score_key] = score
                # **Changed feedback field name to feedback_sa_n**
                feedback_key = f'feedback_sa_{q_num}'
                article[feedback_key] = 'Emotion and sentiment analysis completed successfully.'
            else:
                # Assign default values in case of failure
                for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "negative_sentiment", "positive_sentiment"]:
                    score_key = f"{emotion}_{q_num}"
                    article[score_key] = 0
                # **Changed feedback field name to feedback_sa_n**
                feedback_key = f'feedback_sa_{q_num}'
                article[feedback_key] = 'Failed to analyze emotion and sentiment.'

            # Log the results
            logging.info(f"Article {idx + 1}, Quotation {q_num} Results:\n"
                         f"Joy_{q_num}: {article.get(f'joy_{q_num}', 0)}, Sadness_{q_num}: {article.get(f'sadness_{q_num}', 0)}, "
                         f"Anger_{q_num}: {article.get(f'anger_{q_num}', 0)}, Fear_{q_num}: {article.get(f'fear_{q_num}', 0)}, "
                         f"Surprise_{q_num}: {article.get(f'surprise_{q_num}', 0)}, Disgust_{q_num}: {article.get(f'disgust_{q_num}', 0)}, "
                         f"Trust_{q_num}: {article.get(f'trust_{q_num}', 0)}, Anticipation_{q_num}: {article.get(f'anticipation_{q_num}', 0)}, "
                         f"Negative Sentiment_{q_num}: {article.get(f'negative_sentiment_{q_num}', 0)}, Positive Sentiment_{q_num}: {article.get(f'positive_sentiment_{q_num}', 0)}\n"
                         f"Feedback_sa_{q_num}: {article.get(f'feedback_sa_{q_num}', '')}\n"
                         f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}\n")
            print(f"Article {idx + 1}, Quotation {q_num} - Emotion and Sentiment Ratings Assigned.\n"
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
    print(f"Total articles loaded: {len(articles)}")
    print(f"Articles already processed (with emotion and sentiment ratings): {num_already_processed}")
    print(f"Articles to be processed: {num_to_process}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Cost: ${total_cost:.6f}")
    print("--------------------------\n")

    # Log summary
    logging.info("--- Processing Summary ---")
    logging.info(f"Total articles loaded: {len(articles)}")
    logging.info(f"Articles already processed (with emotion and sentiment ratings): {num_already_processed}")
    logging.info(f"Articles to be processed: {num_to_process}")
    logging.info(f"Total Input Tokens: {total_input_tokens}")
    logging.info(f"Total Output Tokens: {total_output_tokens}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    logging.info("--------------------------\n")

if __name__ == "__main__":
    main()
