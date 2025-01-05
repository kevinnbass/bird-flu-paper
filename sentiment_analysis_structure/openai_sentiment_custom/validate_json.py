import json
import sys

def validate_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"JSON file '{file_path}' is valid.")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSON file syntax.")
    parser.add_argument('json_file', help="Path to the JSON file to validate.")
    args = parser.parse_args()

    validate_json(args.json_file)
