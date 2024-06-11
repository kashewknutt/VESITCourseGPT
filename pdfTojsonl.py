import PyPDF2
import json
import os

input_directory = 'DataSource'  # Folder containing PDFs
output_file = 'pdf_data.jsonl'   # Output JSONL file

def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
    return text

all_data = []

for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(input_directory, filename)
        pdf_text = pdf_to_text(filepath)
        pdf_data = {
            'filename': filename,
            'text': pdf_text
        }
        all_data.append(pdf_data)

with open(output_file, 'w', encoding='utf-8') as jsonl_file:
    for entry in all_data:
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"All PDF files from '{input_directory}' have been successfully converted to '{output_file}'")
