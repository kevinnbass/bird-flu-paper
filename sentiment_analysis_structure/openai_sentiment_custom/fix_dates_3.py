import json
from datetime import datetime

def transform_date_huffpost(date_str):
    """
    Convert a date in M(M)/D(D)/YYYY format to YYYY-MM-DD (zero-padded).
    Example: '1/9/2020' -> '2020-01-09'
    """
    dt = datetime.strptime(date_str, '%m/%d/%Y')
    return dt.strftime('%Y-%m-%d')

def transform_date_forbes(date_str):
    """
    Convert a date in D(D) Month YYYY format to YYYY-MM-DD (zero-padded).
    Example: '29 August 2023' -> '2023-08-29'
    """
    dt = datetime.strptime(date_str, '%d %B %Y')
    return dt.strftime('%Y-%m-%d')

def main():
    input_file = 'processed_all_articles_fixed.jsonl'
    output_file = 'processed_all_articles_fixed_2.jsonl'

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Parse each JSON line
            data = json.loads(line)

            # If there's a field 'media_outlet' and a field 'date', transform accordingly
            if 'media_outlet' in data and 'date' in data:
                if data['media_outlet'] == 'HuffPost':
                    data['date'] = transform_date_huffpost(data['date'])
                elif data['media_outlet'] == 'Forbes':
                    data['date'] = transform_date_forbes(data['date'])
            
            # Write the updated record to the new file
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
