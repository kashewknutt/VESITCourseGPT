import torch
from scraper import create_rag_components
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset
from ragTraining import pdf_data

# Load the dataset
jsonl_file_path = 'RAGdata.json'
dataset = load_dataset('json', data_files=jsonl_file_path, split='train')

# Initialize the tokenizer and model
model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
model = RagTokenForGeneration.from_pretrained(model_name)

# Create RagTokenizer and RagRetriever
tokenizer, retriever = create_rag_components(pdf_data)

# Function to chat with the chatbot
def chat_with_chatbot(query, tokenizer, retriever, generator):
    inputs = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
    search_results = retriever(inputs["input_ids"], return_tensors="pt", n_docs=5)
    generator_input = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "doc_scores": search_results["doc_scores"],
        "doc_ids": search_results["doc_ids"],
    }
    generated = generator.generate(**generator_input, max_length=200)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)

# Main function
def main():
    # Implement the chatbot interface
    print("Chatbot: Hi! I'm your college syllabus chatbot. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        else:
            responses = chat_with_chatbot(user_input, tokenizer, retriever, model)
            print("Chatbot:", responses[0])

if __name__ == "__main__":
    main()
