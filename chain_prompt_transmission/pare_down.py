import json

input_filename = "transmission_chain_7.jsonl"
output_filename = "transmission_chain_7_cleaned.jsonl"

with open(input_filename, "r", encoding="utf-8") as infile, \
     open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        # Strip any extra whitespace/newlines and skip empty lines
        line = line.strip()
        if not line:
            continue
        
        # Parse the line into a Python dictionary
        data = json.loads(line)

        # Remove the 'contextualize' field if it exists
        if "contextualize" in data:
            del data["contextualize"]

        # Remove the 'validation_metadata' field if it exists
        if "validation_metadata" in data:
            del data["validation_metadata"]

        # Write the modified data back as JSONL
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
