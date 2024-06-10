import openai

# Set your OpenAI API key
openai.api_key = 'your_api_key'

# Upload your JSONL file to OpenAI (this returns a file ID)
file_id = openai.File.create(file=open('data.jsonl'), purpose='fine-tune')['id']

# Fine-tune the GPT-3.5-turbo model
fine_tune_response = openai.FineTune.create(
    training_file=file_id,
    model="gpt-3.5-turbo",
    n_epochs=4,
    learning_rate_multiplier=0.1
)

# Monitor the fine-tuning process
print(fine_tune_response)
