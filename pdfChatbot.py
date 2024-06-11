import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# Load the fine-tuned models and tokenizer
tokenizer_combined = AutoTokenizer.from_pretrained("pdf-tuned-model-combined")
model_combined = AutoModelForCausalLM.from_pretrained("pdf-tuned-model-combined")

tokenizer_pdf = AutoTokenizer.from_pretrained("pdf-tuned-model-pdf")
model_pdf = AutoModelForCausalLM.from_pretrained("pdf-tuned-model-pdf")

def chatbot_prompt():
    print("Welcome to the College Syllabus Chatbot!")
    print("Please type your question or type 'exit' to quit.")

def generate_response(input_text, tokenizer, model):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=3, temperature=0.7, do_sample=True)
    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    return responses

def post_process_responses(responses):
    response_count = defaultdict(int)
    filtered_responses = []
    for response in responses:
        if response not in response_count:
            filtered_responses.append(response)
        response_count[response] += 1
    return filtered_responses

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
            tokenizer = tokenizer_pdf
            model = model_pdf
        else:
            tokenizer = tokenizer_combined
            model = model_combined

        responses = generate_response(user_input, tokenizer, model)
        filtered_responses = post_process_responses(responses)

        print("Chatbot:")
        for i, response in enumerate(filtered_responses, start=1):
            print(f"{i}. {response}")
