import json
import os

input_directory = 'jsonTrainers'
output_file = 'combined_data.jsonl'

all_data = []

for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        filepath = os.path.join(input_directory, filename)

        with open(filepath, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            all_data.extend(data)

with open(output_file, 'w', encoding='utf-8') as jsonl_file:
    for entry in all_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"All JSON files from '{input_directory}' have been successfully combined into '{output_file}'")
