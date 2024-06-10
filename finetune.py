import openai
from openai import OpenAI
import time

# Import the API key from the secretKey module
from secretKey import OPENAI_API_KEY


#create the openai client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the path to your JSONL file
jsonl_file_path = 'combined_data.jsonl'

# Upload the file to OpenAI using the new API
print("Uploading file...")
response = client.files.create(
    file=open(jsonl_file_path, 'rb'),
    purpose='fine-tune'
)
file_id = response.id
print(f"File uploaded successfully. File ID: {file_id}")

# Create a fine-tuning job using the new API
print("Creating fine-tuning job...")
fine_tune_response = client.fine_tuning.jobs.create(
    training_file=file_id,
    model='gpt-3.5-turbo'
)
fine_tune_id = fine_tune_response['id']
print(f"Fine-tuning job created. Fine-tune ID: {fine_tune_id}")

# Function to check the status of the fine-tuning job
def check_fine_tune_status(fine_tune_id):
    response = client.FineTune.retrieve(id=fine_tune_id)
    return response['status']

# Polling the status of the fine-tuning job
print("Checking fine-tuning status...")
status = check_fine_tune_status(fine_tune_id)
while status not in ['succeeded', 'failed']:
    print(f"Status: {status}. Checking again in 5 seconds...")
    time.sleep(5)
    status = check_fine_tune_status(fine_tune_id)

# Output the final status of the fine-tuning job
if status == 'succeeded':
    print("Fine-tuning completed successfully!")
else:
    print("Fine-tuning failed. Please check the job details for more information.")
