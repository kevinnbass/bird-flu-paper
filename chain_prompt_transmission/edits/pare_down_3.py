import json

input_filename = "transmission_chain_7.jsonl"
output_filename = "transmission_chain_7_cleaned.jsonl"

# Any top-level fields you want to remove entirely
fields_to_remove_top_level = [
    "validation_metadata",
    "context_added",    # If it appears at the top level
    "unchanged"         # If it appears at the top level
]

with open(input_filename, "r", encoding="utf-8") as infile, \
     open(output_filename, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
        
        # Parse the JSON object
        data = json.loads(line)
        
        # 1. Remove certain fields if they exist at the top level
        for field in fields_to_remove_top_level:
            if field in data:
                del data[field]
        
        # 2. If "contextualize" exists, remove "context_added" and "unchanged" from it
        if "contextualize" in data and isinstance(data["contextualize"], dict):
            # Remove "context_added" if it exists
            if "context_added" in data["contextualize"]:
                del data["contextualize"]["context_added"]
            
            # Remove "unchanged" if it exists
            if "unchanged" in data["contextualize"]:
                del data["contextualize"]["unchanged"]
        
        # Write out the modified line
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
