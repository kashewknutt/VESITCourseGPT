import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

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

# Split datasets into train and validation sets (80-20 split)
def split_dataset(dataset, split_ratio=0.2):
    dataset = dataset.train_test_split(test_size=split_ratio)
    return dataset['train'], dataset['test']

# Process and split combined data
combined_data_tokenized = process_combined_data('combined_data.jsonl')
train_combined, eval_combined = split_dataset(combined_data_tokenized)

# Process and split PDF data
pdf_data_tokenized = process_pdf_data('pdf_data.jsonl')
train_pdf, eval_pdf = split_dataset(pdf_data_tokenized)

# Define training arguments for combined data
# Adjust learning rate and add a scheduler if needed
training_args_combined = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True,
    learning_rate=5e-5,  # You can start with a lower learning rate
)


# Trainer for combined data
trainer_combined = Trainer(
    model=model,
    args=training_args_combined,
    train_dataset=train_combined,
    eval_dataset=eval_combined,  # Evaluation dataset
)

# Train and evaluate combined data
trainer_combined.train()
trainer_combined.evaluate()  # Evaluate after training

# Save the fine-tuned model after training combined data
model.save_pretrained("pdf-tuned-model-combined")
tokenizer.save_pretrained("pdf-tuned-model-combined")
print("Conversation tuning saved to pdf-tuned-model-combined")

# Define training arguments for PDF data
training_args_pdf = TrainingArguments(
    output_dir='./results_pdf',
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir='./logs_pdf',
    logging_steps=500,
    load_best_model_at_end=True,
    learning_rate=5e-5,  # You can start with a lower learning rate
)


# Trainer for PDF data
trainer_pdf = Trainer(
    model=model,
    args=training_args_pdf,
    train_dataset=train_pdf,
    eval_dataset=eval_pdf,  # Evaluation dataset
)

# Train and evaluate PDF data
trainer_pdf.train()
trainer_pdf.evaluate()  # Evaluate after training

# Save the fine-tuned model after training PDF data
model.save_pretrained("pdf-tuned-model-pdf")
tokenizer.save_pretrained("pdf-tuned-model-pdf")
print("PDF tuning saved to pdf-tuned-model-pdf")
