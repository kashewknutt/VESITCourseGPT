import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Define the chatbot function
def chatbot(query):
    inputs = tokenizer.encode("User: " + query, return_tensors="pt", max_length=512, truncation=True)
    response_ids = model.generate(inputs, max_length=1000, num_return_sequences=1, temperature=0.9)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Implement the command-line interface
print("Chatbot: Hi! I'm your college syllabus chatbot. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    else:
        response = chatbot(user_input)
        print("Chatbot:", response)
