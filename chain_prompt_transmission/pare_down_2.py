import json

input_filename = "transmission_chain_7.jsonl"
output_filename = "transmission_chain_7_cleaned.jsonl"

fields_to_remove = [
    "validation_metadata",
    "context_added",
    "unchanged"
]

with open(input_filename, "r", encoding="utf-8") as infile, \
     open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        # Strip any extra whitespace/newlines and skip empty lines
        line = line.strip()
        if not line:
            continue
        
        # Parse the line into a Python dictionary
        data = json.loads(line)

        # Remove specified fields if they exist
        for field in fields_to_remove:
            if field in data:
                del data[field]

        # Write the modified data back as JSONL
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
