# Fine-Tuning GPT-3.5 for College Syllabus Chatbot

This repository contains the code and instructions to fine-tune the GPT-3.5 model for creating a chatbot that answers questions related to a college syllabus.

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
    - JSONL format is preferred for fine-tuning GPT-3.5.

3. **Set Up OpenAI API Key**
    - Create a Python file (e.g., `secretKey.py`) to store your OpenAI API key.
    - Add your API key to the file:
        ```python
        OPENAI_API_KEY = "your-api-key"
        ```
4. **Fine-Tune GPT-3.5**
    - Use the OpenAI Python library to fine-tune the GPT-3.5 model.
    - Upload the JSONL file using the new API.
    - Create a fine-tuning job with the uploaded file and specify the model.

5. **Monitor Fine-Tuning Job**
    - Check the status of the fine-tuning job periodically.
    - Wait for the job to complete successfully.

## Files Included

- `jsonTrainers/`: Folder containing the structured JSON files.
- `jsonTojsonl.py`: Python script to convert JSON files to JSONL format.
- `secretKey.py`: Python file to store the OpenAI API key.
- `finetune.py`: Python script to fine-tune the GPT-3.5 model.

## Usage

1. Clone this repository to your local machine.
2. Place your structured JSON files in the `jsonTrainers/` folder.
3. Run the `jsonTojsonl.py` script to convert JSON to JSONL.
4. Set up your OpenAI API key in `secretKey.py`.
5. Run the `finetune.py` script to initiate the fine-tuning process.

## Notes

- Ensure you have the necessary permissions and resources to fine-tune the GPT-3.5 model.
- Monitor the fine-tuning job closely and handle any errors or failures accordingly.

## Contributors

- Rajat Disawal (github.com/kashewknutt)
