import os
import openai
import json
import random
import yaml
from tqdm import tqdm
import argparse

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Set OpenAI API key
def set_openai_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

def load_text_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 各行をリストに格納（空行は除外）
        return [line.strip() for line in file.readlines() if line.strip()]

# Function to query GPT and save the response to a file
def query_gpt_chat(input_str, filename, model, temperature):
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
        response_text = response.choices[0].message.content
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

    # Extract global settings
    json_make = config['json_make']
    model = json_make['model']
    temperature = json_make['temperature']
    output_base_path = json_make['output_base_path']
    chartlist = json_make['datalist']
    data_file_path = json_make['data_file_path']

    # Process each topic
    for data in chartlist:
        topic_file = os.path.join(data_file_path, f"{data.replace(' ', '_')}.txt")

        topiclist = load_text_to_list(topic_file)

        # Ensure the output directory exists
        data_output_path = os.path.join(output_base_path, data.replace(' ', '_'))
        if not os.path.exists(data_output_path):
            os.makedirs(data_output_path)

        # Generate JSONs
        with tqdm(topiclist, desc=f"Creating JSON {data}", unit="chart type") as pbar:
            for i, topic in enumerate(pbar):
                # Load example JSON files
                example_json_directory = json_make['example_json_directory']
                chart_json_data = load_json_files(
                    os.path.join(example_json_directory, data.replace(' ', '_')),
                    num_samples=2
                )
                topic = topic.strip()

                # Select two random examples
                example1, example2 = random.sample(chart_json_data, 2)
                example1 = json.dumps(example1)
                example2 = json.dumps(example2)

                # Construct the prompt
                prompt = (
                    f"Example1: {example1}, Example2: {example2}. Based on these examples, create one JSON file to show {topic} for creating a {data} chart. You will output all responses in JSON format."
                )
                # Define output file name
                output_file = os.path.join(data_output_path, f"{i}.json")
                query_gpt_chat(prompt, output_file, model, temperature)

    print(f"All JSONs processed and saved in {output_base_path}")

if __name__ == "__main__":
    # Use argparse to specify the configuration path
    parser = argparse.ArgumentParser(description="Generate JSON files using OpenAI API.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
