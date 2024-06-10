import os
import json
from PyPDF2 import PdfReader

# Function to extract text from PDF documents
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text.strip()

# Function to structure PDF data into JSON format
def structure_pdf_data(pdf_folder):
    pdf_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            # Extract metadata if needed
            metadata = {
                "filename": filename,
                "text": text
            }
            pdf_data.append(metadata)
    return pdf_data

# Combine JSON data from PDFs with existing JSON data
def combine_json_data(json_folder, pdf_data):
    combined_data = []
    # Load JSON data from existing JSON files
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            with open(os.path.join(json_folder, filename), 'r') as file:
                existing_data = json.load(file)
                combined_data.extend(existing_data)
    # Append PDF data to existing JSON data
    combined_data.extend(pdf_data)
    return combined_data

# Paths to the PDF folder and JSON folder
pdf_folder = 'DataSource'
json_folder = 'jsonTrainers'

# Extract text from PDFs and structure into JSON format
pdf_data = structure_pdf_data(pdf_folder)

# Combine PDF data with existing JSON data
combined_data = combine_json_data(json_folder, pdf_data)

# Save the combined data to a JSON file
output_file = 'RAGdata.json'
with open(output_file, 'w') as file:
    json.dump(combined_data, file, indent=4)

print("Data successfully combined and saved to:", output_file)
