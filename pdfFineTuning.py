import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Initialize the tokenizer and model
model_name = "gpt2"  # Replace with "gpt-neo-125M" or another model if desired
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Define function to handle combined data JSONL file
def process_combined_data(jsonl_file):
    dataset = load_dataset('json', data_files=jsonl_file, split='train')

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
        
        return {
            "input_ids": tokenized_texts["input_ids"],
            "labels": tokenized_texts["input_ids"]  # Assuming autoregressive generation (predicting the next token)
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset

# Define function to handle PDF data JSONL file
def process_pdf_data(jsonl_file):
    dataset = load_dataset('json', data_files=jsonl_file, split='train')

    def tokenize_function(example):
        text = example['text']
        tokenized_texts = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        
        return {
            "input_ids": tokenized_texts["input_ids"],
            "labels": tokenized_texts["input_ids"]  # Assuming autoregressive generation (predicting the next token)
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset

# Tokenize and fine-tune combined data
combined_data_tokenized = process_combined_data('combined_data.jsonl')
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_data_tokenized,
)
trainer.train()

# Save the fine-tuned model after training combined data
model.save_pretrained("pdf-tuned-model-combined")
tokenizer.save_pretrained("pdf-tuned-model-combined")
print("conversation tuning saved to pdf-tuned-model-combined")

# Tokenize and fine-tune PDF data
pdf_data_tokenized = process_pdf_data('pdf_data.jsonl')
training_args = TrainingArguments(
    output_dir='./results_pdf',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs_pdf',
    logging_steps=500,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pdf_data_tokenized,
)
trainer.train()

# Save the fine-tuned model after training PDF data
model.save_pretrained("pdf-tuned-model-pdf")
tokenizer.save_pretrained("pdf-tuned-model-pdf")
print("pdf tuning saved to pdf-tuned-model-pdf")
