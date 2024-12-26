import json
import yaml
import random
import os
from tqdm import tqdm
import argparse

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def add_colors_to_charts(json_directory, colors_file, output_directory):
    """
    Adds colors to chart JSON files based on the chart type, using colors from a CSV file.
    Args:
        json_directory (str): Path to the directory containing JSON files.
        colors_file (str): Path to the CSV file containing color definitions.
        output_directory (str): Path to the directory where updated JSON files with color information will be saved.
    Returns:
        int: Total number of files processed.
    """
    total = 0

    # Read color information from the CSV file (left column only)
    with open(colors_file, 'r') as file:
        colors = [line.split(',')[0].strip() for line in file.readlines()]

    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Process JSON files in the specified directory and its subdirectories
    for root, dirs, files in os.walk(json_directory):
        for chart_folder in dirs:
            chart_output_directory = os.path.join(output_directory, chart_folder)
            os.makedirs(chart_output_directory, exist_ok=True)

            # Process files in the current chart folder
            folder_path = os.path.join(root, chart_folder)
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            for filename in tqdm(json_files, desc=f"Adding color information in {chart_folder}", unit="file"):
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    # Handle list format in JSON data
                    if isinstance(data, list):
                        data = data[0]

                    if 'models' not in data or not isinstance(data['models'], list):
                        print(f"Skipping {filename}: 'models' key not found or is not a list.")
                        continue


                    # Apply colors based on chart type
                    if chart_folder == 'Diverging_bar_chart':
                        data = add_colors_to_diverging_bar_chart(data, colors)
                    elif chart_folder == 'Horizontal_bar_chart':
                        data = add_colors_to_horizontal_bar_chart(data, colors)
                    elif chart_folder == 'Vertical_bar_chart':
                        data = add_colors_to_vertical_bar_chart(data, colors)
                    elif chart_folder in ['Horizontal_grouped_bar_chart', 'Vertical_grouped_bar_chart',
                                          'Horizontal_stacked_bar_chart', 'Vertical_stacked_bar_chart']:
                        data = add_colors_to_h_grouped_bar_chart(data, colors)
                    elif chart_folder == 'Basic_pie_chart':
                        data = add_color_to_pie_chart(data, colors)
                    elif chart_folder in ['Line_plot', 'Scatter_plot']:
                        data = add_color_to_line_chart_models(data, colors)
                    else:
                        continue

                    # Save the updated JSON file
                    chart_output_directory = os.path.join(output_directory, chart_folder)
                    os.makedirs(chart_output_directory, exist_ok=True)

                    new_file_path = os.path.join(chart_output_directory, os.path.basename(file_path))
                    with open(new_file_path, 'w') as file:
                        total += 1
                        json.dump(data, file, indent=4)

                except json.JSONDecodeError:
                    print(f"Skipping {filename}: JSON decoding error.")
                    continue
    return total

def add_colors_to_diverging_bar_chart(data, colors):
    """Adds colors to a diverging bar chart."""
    if len(data['models']) > 2:
        data['models'] = data['models'][:2]

    for model in data['models']:
        try:
            color = random.choice(colors)
            if isinstance(model['y'], list):
                color_list = [color] * len(model['y'])
                model['colors'] = color_list
        except KeyError:
            print("Skipping a model due to missing 'y' key.")
            continue
        except TypeError as e:
            print(f"Skipping model due to type error: {e}")
            continue
        except Exception as e:
            print(f"Skipping model due to unexpected error: {e}")
            continue

    return data

def add_colors_to_horizontal_bar_chart(data, colors):
    """Adds colors to a horizontal bar chart."""
    if len(data['models']) > 1:
        data['models'] = [data['models'][0]]

    for model in data['models']:
        try:
            use_same_color = random.choice([True, False])
            if isinstance(model['y'], list):
                color_list = [random.choice(colors) for _ in model['y']] if not use_same_color else [random.choice(colors)] * len(model['y'])
                model['colors'] = color_list
        except KeyError:
            print("Skipping a model due to missing 'y' key.")
            continue
        except TypeError as e:
            print(f"Skipping model due to type error: {e}")
            continue
        except Exception as e:
            print(f"Skipping model due to unexpected error: {e}")
            continue

    return data

def add_colors_to_vertical_bar_chart(data, colors):
    """Adds colors to a vertical bar chart."""
    if len(data['models']) > 1:
        data['models'] = [data['models'][0]]

    for model in data['models']:
        try:
            use_same_color = random.choice([True, False])
            if isinstance(model['x'], list):
                color_list = [random.choice(colors) for _ in model['x']] if not use_same_color else [random.choice(colors)] * len(model['x'])
                model['colors'] = color_list
        except KeyError:
            print("Skipping a model due to missing 'x' key.")
            continue
        except TypeError as e:
            print(f"Skipping model due to type error: {e}")
            continue
        except Exception as e:
            print(f"Skipping model due to unexpected error: {e}")
            continue

    return data

def add_colors_to_h_grouped_bar_chart(data, colors):
    """Adds colors to a grouped or stacked bar chart."""
    for model in data['models']:
        try:
            if isinstance(model['x'], list):
                color_list = [random.choice(colors)] * len(model['x'])
                model['colors'] = color_list
        except KeyError:
            print("Skipping a model due to missing 'x' key.")
            continue

    return data

def add_color_to_pie_chart(data, colors):
    """Adds colors to a pie chart."""
    for model in data['models']:
        try:
            model['colors'] = random.choice(colors)
        except KeyError:
            print("Skipping a model due to missing 'value' key.")
            continue
        except TypeError as e:
            print(f"Skipping model due to type error: {e}")
            continue
        except Exception as e:
            print(f"Skipping model due to unexpected error: {e}")
            continue

    return data

def add_color_to_line_chart_models(data, colors):
    """Adds colors to a line or scatter chart."""
    for model in data['models']:
        try:
            model['colors'] = random.choice(colors)
        except KeyError:
            print("Skipping a model due to missing 'x' or 'y' keys.")
            continue
        except TypeError as e:
            print(f"Skipping model due to type error: {e}")
            continue
        except Exception as e:
            print(f"Skipping model due to unexpected error: {e}")
            continue

    return data

# Main function
def main(config_path):
    # Load configuration
    config = load_config(config_path)
    # Load settings from config
    json_directory = config['add_color']['json_directory']
    colors_file = config['add_color']['colors_file']
    output_directory = config['add_color']['output_directory']

    # Process charts and add colors
    total = add_colors_to_charts(json_directory, colors_file, output_directory)

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
