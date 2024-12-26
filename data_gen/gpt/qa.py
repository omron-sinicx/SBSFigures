import os
import openai
import json
import random
import yaml
from tqdm import tqdm
import glob
import re
import argparse

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Set OpenAI API key
def set_openai_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Function to query GPT and save the response to a file
def query_gpt_chat(input_str, filename, temperature, model):
    try:
        # Generate response using OpenAI Chat API
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_str}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        response_text = response.choices[0].message.content.strip()
        response_json = json.loads(response_text)
    except Exception as e:
        response_json = f"Error: {e}"

    # JSON形式で保存
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(response_json, file, ensure_ascii=False, indent=4)

    return response_json

# Function to load a specified number of JSON files from a directory
def load_json_files(directory, num_samples):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    selected_files = random.sample(json_files, min(len(json_files), num_samples))
    samples = []
    for file in selected_files:
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as json_file:
            samples.append(json.load(json_file))
    return samples

# Main script logic
def main(config_path):
    # Load the configuration
    config = load_config(config_path)

    # Set OpenAI API key
    set_openai_api_key(config['api_key'])

    # Define directories and parameters
    base_directory = config['qa']['base_directory']
    output_directory = config['qa']['output_directory']
    example_directory = config['qa']['example_directory']
    chart_types = config['qa']['datalist']
    model = config['qa']['model']
    temperature = config['qa']['temperature']

    for chart in chart_types:
        # Set chart-specific directories
        chart_directory = os.path.join(base_directory, chart.replace(' ', '_'))
        chart_output_directory = os.path.join(output_directory, chart.replace(' ', '_'))

        if "bar" in chart:
            sample_directory = os.path.join(example_directory, "bar")
        elif "pie" in chart:
            sample_directory = os.path.join(example_directory, "pie")
        elif "plot" in chart:
            sample_directory = os.path.join(example_directory, "plot")

        if not os.path.exists(chart_output_directory):
            os.makedirs(chart_output_directory)

        # Load JSON files from the specified directory
        json_files = glob.glob(os.path.join(chart_directory, "*.json"))

        # Process each JSON file
        for file_path in tqdm(json_files, desc=f"Creating {chart} QA", unit="chart type"):
            base_name = os.path.basename(file_path)
            file_number = re.search(r"(\d+)", base_name).group(0)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                data_str = json.dumps(data)

            # Load examples
            example1, example2 = load_json_files(sample_directory, num_samples=2)

            # Merge the two examples into a single JSON object
            merged_examples = {
                "qa": [example1, example2]
            }

            merged_examples_str = json.dumps(merged_examples)

            # Generate prompt
            prompt = (
                f"Example: {merged_examples_str}."
                f"Based on these examples, create relevant questions and their answers (labels) "
                f"based on the following JSON data. The queries should be relevant to the data categories, "
                f"values, and colors specified in the JSON structure. The labels should correctly answer "
                f"the queries based on the data. {data_str}."
            )

            # Define output file name
            file_name = os.path.join(chart_output_directory, f'{file_number}.json')

            # Query GPT and save the output
            query_gpt_chat(prompt, file_name, temperature, model)

        print("All prompts processed and saved.")

if __name__ == "__main__":
    # Use argparse to get the configuration path from the command line
    parser = argparse.ArgumentParser(description="Process JSON files and generate GPT responses.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Pass the config path to the main function
    main(args.config)


