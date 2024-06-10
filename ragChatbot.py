import torch
from transformers import RagSequenceForGeneration
from scraper import scrape_documents, create_rag_components

# Load the fine-tuned model for generation
model_path = "./fine-tuned-model"
generator = RagSequenceForGeneration.from_pretrained(model_path)

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
    # XValue addresses where documents are located
    xvalue_addresses = [
        'https://vesit.ves.ac.in/departments/DSAI/po',
        'https://vesit.ves.ac.in/departments/DSAI/syllabus',
        'https://vesit.ves.ac.in/departments/timetable/DSAI#current'
    ]
    
    # Scrape documents from XValue addresses
    documents = scrape_documents(xvalue_addresses)
    
    # Create RagTokenizer and RagRetriever
    tokenizer, retriever = create_rag_components(documents)
    
    # Implement the chatbot interface
    print("Chatbot: Hi! I'm your college syllabus chatbot. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        else:
            responses = chat_with_chatbot(user_input, tokenizer, retriever, generator)
            print("Chatbot:", responses[0])

if __name__ == "__main__":
    main()
