import os
import json

from openai_compat import create_chat_completion, create_client, extract_text_content, get_default_model

class Summary_Module:
    def __init__(self, api_key, base_url=None, model=None):
        """
        Initialize the Summary object with the OpenAI API Key.

        Args:
            api_key (str): OpenAI 的 API Key
        """
        self.api_key = api_key
        self.client = create_client(api_key, base_url=base_url)
        self.model = model or get_default_model()

    def summarize(self, input_file, output_file, prompt, field):
        """
        Summarize the content of a specified field in a JSON file using OpenAI ChatCompletion.

        Args:
            input_file (str): input file path
            output_file (str): output file path
            field (str): field name to summarize

        Returns:
            str: summarized text
        """
        with open(input_file, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as file:
                output_data = json.load(file)
        else:
            output_data = {}

        if field not in input_data:
            print(f"field '{field}' not found in the input data.")
            return
        content = input_data[field]

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please help me summarize the information."},
            # {"role": "user", "content": f"{content}\n {prompt} \nAnswer with only one word (Yes, No or Uncertain)"}
            {"role": "user", "content": f"{content}\n {prompt} \nAnswer with only one word (Yes or No)"}
        ]

        completion = create_chat_completion(
            self.client,
            messages=messages,
            model=self.model,
        )
        summary_text = extract_text_content(completion)

        output_data[field] = summary_text

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, indent=4)

        return summary_text
