import json
from collections import Counter
import os
import sys

def count_media_outlets(json_file_path):
    # Check if the file exists
    if not os.path.isfile(json_file_path):
        print(f"Error: File '{json_file_path}' does not exist.")
        sys.exit(1)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # Initialize a counter for media outlets
    media_outlet_counter = Counter()
    total_entries = 0

    # Iterate through each entry in the JSON data
    for entry in data:
        total_entries += 1
        media_outlet = entry.get('media_outlet', 'Unknown')
        media_outlet_counter[media_outlet] += 1

    # Display the counts
    print("Counts of entries per 'media_outlet':")
    for outlet, count in media_outlet_counter.items():
        print(f" - {outlet}: {count}")

    print(f"\nTotal number of entries: {total_entries}")

if __name__ == "__main__":
    json_file = 'all_articles_bird_flu_project_for_sa.json'
    count_media_outlets(json_file)
