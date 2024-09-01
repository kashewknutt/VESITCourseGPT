import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
jsonl_file_path = 'combined_data.jsonl'
dataset = load_dataset('json', data_files=jsonl_file_path, split='train')

# Inspect the dataset structure
print("Dataset sample:", dataset[0])

# Initialize the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Prepare the conversations into the right format for tokenization
def format_conversation(conversation):
    formatted_conversation = ""
    for message in conversation:
        role = message['role']
        content = message['content']
        formatted_conversation += f"{role}: {content}\n"
    return formatted_conversation

def tokenize_function(examples):
    conversations = examples['messages']
    formatted_texts = [format_conversation(conversation) for conversation in conversations]
    tokenized_texts = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=512)
    
    # Extract input_ids and labels from the tokenized_texts
    return {
        "input_ids": tokenized_texts["input_ids"],
        "labels": tokenized_texts["input_ids"]  # Assuming autoregressive generation (predicting the next token)
    }


# Apply tokenization (batched=True processes multiple examples at once)
print("Tokenizing the dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning the model
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start the training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
print("Model saved successfully!")
