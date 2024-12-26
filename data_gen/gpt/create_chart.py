import os
import importlib.util
import json
import random
import yaml
from tqdm import tqdm
import argparse

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def get_random_font_and_size(font_file):
    with open(font_file, 'r') as file:
        fonts = file.readlines()
    font_name = random.choice(fonts).strip()
    font_size = random.randint(11, 13)
    return font_name, font_size

def process_chart_data(json_string, json_save_path, basename_without_ext, selected_script, font_file_path, figure_save_path):
    font_name, font_size = get_random_font_and_size(font_file_path)
    
    spec = importlib.util.spec_from_file_location("plotting_module", selected_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    save_path = f'{figure_save_path}/{basename_without_ext}.png'

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)

    try:
        module.plot_chart(json_string, save_path, json_save_path, font_name, font_size)
    except KeyError as e:
        print(f"Skipping chart due to KeyError in {json_save_path}: {e}")
    except Exception as e:
        print(f"Skipping chart due to unexpected error in {json_save_path}: {e}")


# Main script logic
def main(config_path):
    # Load the configuration
    config = load_config(config_path)
    font_file_path = config['create_chart']['font_txt']
    python_scripts_dir = config['create_chart']['python_scripts_dir']
    json_base_dir = config['create_chart']['json_directly']
    json_save_base_dir= config['create_chart']['json_output_directly']
    figure_output_directly = config['create_chart']['figure_output_directly']

    subdirs = [folder for folder in os.listdir(json_base_dir) if os.path.isdir(os.path.join(json_base_dir, folder))]

    for subdir in subdirs:
        subdir_path = os.path.join(json_base_dir, subdir)
        subdir_save_path = os.path.join(json_save_base_dir, subdir)
        figure_save_path = os.path.join(figure_output_directly, subdir)
        os.makedirs(subdir_save_path, exist_ok=True)
        os.makedirs(figure_save_path, exist_ok=True)
        if os.path.isdir(subdir_path):
            script_file_for_subdir = os.path.join(python_scripts_dir, subdir + ".py")
            with tqdm(os.listdir(subdir_path), desc=f"Creating figures {subdir}", unit="chart type") as pbar:
                for json_file in pbar:
                    if json_file.endswith('.json'):
                        json_path = os.path.join(subdir_path, json_file)
                        json_save_path = os.path.join(subdir_save_path, json_file)
                        basename_without_ext = os.path.splitext(os.path.basename(json_path))[0]
                        try:
                            with open(json_path, 'r') as file:
                                json_input = json.load(file)

                            json_string = json.dumps(json_input)
                            selected_script = script_file_for_subdir
                            process_chart_data(json_string, json_save_path, basename_without_ext, selected_script, font_file_path, figure_save_path)
                        except (json.decoder.JSONDecodeError, KeyError) as e:
                            print(f"Skipping file due to error in {json_path}: {e}")
                            continue
                        except Exception as e:
                            print(f"Skipping file due to unexpected error in {json_path}: {e}")
                            continue


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
