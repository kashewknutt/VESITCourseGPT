# Fine-Tuning GPT for College Syllabus Chatbot

This repository contains the code and instructions to fine-tune the GPT model for creating a chatbot that answers questions related to a college syllabus.

## Steps

1. **Data Collection and Structuring**
    - Create JSON files containing question-answer pairs related to the college syllabus.
    - Structure each JSON file with the following format:
        ```json
        [
            {
                "messages": [
                    {"role": "system", "content": "Description for chatbot"},
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                    ...
                ]
            },
            ...
        ]
        ```

2. **Convert JSON to JSONL**
    - Use a Python script to convert the structured JSON files to a single JSONL file.
    - JSONL format is preferred for fine-tuning GPT.

3. **Fine-Tune GPT**
    - Use the Hugging Face `transformers` library to fine-tune the GPT model.
    - Tokenize the input data and prepare it for training.
    - Fine-tune the model on the prepared dataset.

4. **Monitor Fine-Tuning Process**
    - Check the training progress regularly.
    - Tune hyperparameters if necessary.
    - Save the fine-tuned model once training is complete.

## Files Included

- `DataSource/`: Folder containing the PDF files.
- `jsonTrainers/`: Folder containing the structured JSON files.
- `pdfTojsonl.py`: Python script to convert PDF files to JSONL format.
- `jsonTojsonl.py`: Python script to convert JSON files to JSONL format.
- `finetune.py`: Python script to fine-tune the GPT model using the example conversations.
- `pdfFineTuning.py`: Python script to fine-tune the GPT model using both the conversations and the PDF files.
- `chatbot.py`: Python script containing the code for the chatbot.
- `pdfChatbot.py`: Python script containing the code for the chatbot.

## Usage

### Option 1: Train with JSON Conversations Only

1. Clone this repository to your local machine.
2. Place your structured JSON files in the `jsonTrainers/` folder.
3. Run the `jsonTojsonl.py` script to convert JSON to JSONL.
4. Run the `finetune.py` script to initiate the fine-tuning process.
5. Monitor the training process and save the fine-tuned model once training is complete.
6. Use the `chatbot.py` script to interact with the trained model and ask questions related to the college syllabus.

### Option 2: Train with Both PDFs and JSON Conversations

1. Clone this repository to your local machine.
2. Place your structured JSON files in the `jsonTrainers/` folder and your PDF files in the `DataSource/` folder.
3. Run the `pdfTojsonl.py` script to convert PDF files to JSONL.
4. Run the `jsonTojsonl.py` script to convert JSON files to JSONL format.
5. Run the `pdfFineTuning.py` script to fine-tune the model using both PDFs and JSON conversations.
6. Monitor the training process and save the fine-tuned model once training is complete.
7. Use the `pdfChatbot.py` script to interact with the trained model and ask questions related to the college syllabus.

## Notes

- Ensure you have the necessary permissions and resources to fine-tune the GPT model.
- Adjust hyperparameters and model architecture as needed for your specific use case.
- Monitor the training process closely and handle any errors or failures accordingly.

## Contributors

- Rajat Disawal (github.com/kashewknutt)
