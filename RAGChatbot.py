from transformers import RagTokenizer, DPRReader, RagTokenForGeneration

# Initialize the tokenizer and model
model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
model = RagTokenForGeneration.from_pretrained(model_name)

# Create DPR Reader
reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")

# Function to chat with the chatbot
def chat_with_chatbot(query, tokenizer, reader, generator):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = reader(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    if outputs is None:
        return ["Sorry, I couldn't find relevant information."]
    if "input_ids" not in outputs or "attention_mask" not in outputs:
        return ["Sorry, I couldn't process your query."]
    generated = generator.generate(input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"], max_length=200)
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
            responses = chat_with_chatbot(user_input, tokenizer, reader, model)
            print("Chatbot:", responses[0])

if __name__ == "__main__":
    main()
