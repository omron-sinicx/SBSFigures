import os
import yaml
import openai
from tqdm import tqdm
import argparse
import re

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Set OpenAI API key
def set_openai_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Function to query the model and append the output to a single file
def query_gpt_chat(input_str, output_file, model, temperature):
    try:
        # Perform text generation
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_str}
            ],
            temperature=temperature
        )
        # Extract the text content from the response
        response_text = response.choices[0].message.content.strip()
        
        # Remove numbers and list markers (e.g., "1. ", "2. ")
        cleaned_text = re.sub(r'^\d+\.\s*', '', response_text, flags=re.MULTILINE)

        # Append the generated text directly to the output file
        with open(output_file, 'a', encoding='utf-8') as file:
            file.write(cleaned_text + "\n")

    except Exception as e:
        # Handle any errors (e.g., OpenAI API errors or file write errors)
        raise RuntimeError(f"An error occurred: {e}")

    return response_text

# Main function
def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Set OpenAI API key
    set_openai_api_key(config["api_key"])

    # Iterate over data topics
    data_topics = config["data_topic"]

    # Extract global settings
    output_base_path = data_topics["output_base_path"]
    model = data_topics["model"]
    temperature = data_topics["temperature"]
    num_examples = data_topics["num_examples"] 
    chartlist = data_topics["datalist"]
    num_iterations = data_topics["num_iterations"]

    # Process each chart type
    for data in tqdm(chartlist, desc="Processing charts", unit="chart"):
        tqdm.write(f"Creating data topics: {data}")
        output_dir = os.path.join(output_base_path)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        output_file = os.path.join(output_base_path, f"{data.replace(' ', '_')}.txt")

        # Generate prompts and append results to the file
        for i in range(num_iterations):
            prompt = (
                f"List {num_examples} examples of data that would be appropriate to represent in a {data}. Please answer directly."
            )
            query_gpt_chat(prompt, output_file, model, temperature)

    print(f"All prompts processed and saved in {output_base_path}")

if __name__ == "__main__":
    # Use argparse to specify the configuration path
    parser = argparse.ArgumentParser(description="Generate data topics using OpenAI GPT.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
