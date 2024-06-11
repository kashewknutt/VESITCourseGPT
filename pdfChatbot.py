import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned models and tokenizer
tokenizer_combined = AutoTokenizer.from_pretrained("pdf-tuned-model-combined")
model_combined = AutoModelForCausalLM.from_pretrained("pdf-tuned-model-combined")

tokenizer_pdf = AutoTokenizer.from_pretrained("pdf-tuned-model-pdf")
model_pdf = AutoModelForCausalLM.from_pretrained("pdf-tuned-model-pdf")

def chatbot_prompt():
    print("Welcome to the College Syllabus Chatbot!")
    print("Please type your question or type 'exit' to quit.")

def generate_response(input_text, tokenizer, model):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main chat loop
while True:
    chatbot_prompt()
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    else:
        # Choose the appropriate model based on user input
        if "pdf" in user_input.lower():
            print("Pdf Tokenizer")
            tokenizer = tokenizer_pdf
            model = model_pdf
        else:
            print("Conversation Tokenizer")
            tokenizer = tokenizer_combined
            model = model_combined

        response = generate_response(user_input, tokenizer, model)
        print("Chatbot:", response)
