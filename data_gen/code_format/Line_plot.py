import json
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.lines import Line2D

def extract_data(json_input):
    """Extract data from the JSON input."""
    data = json.loads(json_input)
    return data

def convert_to_float_and_extract_unit(value):
    """Extract the numeric value and handle percentage symbols."""
    value = value.replace(',', '')
    if value.endswith('%'):
        number = float(value.strip('%'))
        return number, '%'
    elif value.endswith('k'):
        number = float(value.strip('k'))
        return number, 'k'
    elif value.startswith('$'):
        number = float(value.strip('$'))
        return number, '$'
    else:
        try:
            return float(value), ''
        except ValueError:
            return None, ''

def plot_chart(json_input, save_path, json_save_path, font_name='Arial', font_size=12):
    """Plot a line chart based on the input JSON with variations."""
    data = extract_data(json_input)

    x_values_list = [model['x'] for model in data['models']]

    y_values_converted = []
    units_list = []
    for model in data['models']:
        converted_values = []
        units = []
        for value in model['y']:
            converted, unit = convert_to_float_and_extract_unit(str(value))
            if converted is not None:
                converted_values.append(converted)
                units.append(unit)
            else:
                print(f"Warning: Conversion failed for value {value}")
                return
        y_values_converted.append(converted_values)
        units_list.append(units)

    y_values_list = np.array(y_values_converted, dtype=float)
    colors = [model['colors'] for model in data['models']]
    names = [model['name'] for model in data['models']]

    title_option = random.choice([True, False])
    legend_option = random.choice([True, False, 'barside'])
    legend_loc_option = random.choice([('upper left', (1, 0.9)), ('lower right', (1, 0)), ('upper right', (1, 1)), ('lower left', (0, 0)), ('upper center', (0.5, 1))])
    label_color_option = random.choice(['same', 'black'])
    value_color_option = random.choice(['same', 'black'])
    axes_option = random.choice([True, False])
    value_labels_option = random.choice([True, False])

    fig, ax = plt.subplots()

    for i, (x_values, y_values, color, name) in enumerate(zip(x_values_list, y_values_list, colors, names)):
        ax.plot(x_values, y_values, label=name if legend_option != 'barside' else None, color=color, marker='o')

        if value_labels_option:
            for x, y in zip(x_values, y_values):
                unit = units_list[i][0]
                if y.is_integer():
                    text_format = f"{y:.0f}"
                else:
                    text_format = f"{y:.1f}"
                text = text_format + unit if unit.strip() else text_format
                ax.text(x, y, text, color=color if value_color_option == 'same' else 'black', fontsize=font_size-2, verticalalignment='bottom')

        if legend_option == 'barside':
            label_position = random.choice(['left', 'center', 'right'])
            random_index = random.randint(0, len(x_values) - 1)
            x, y = x_values[random_index], y_values[random_index]
            label_text = f'{name}'
            label_color = color if label_color_option == 'same' else 'black'
            ha_mapping = {'left': 'right', 'center': 'center', 'right': 'left'}
            ax.text(x, y, label_text, color=label_color, ha=ha_mapping[label_position], fontsize=font_size, verticalalignment='bottom')

    ax.set_xlabel(data['x_axis'], fontsize=font_size, fontname=font_name)
    if axes_option or not value_labels_option:
        ax.set_ylabel(data['y_axis'], fontsize=font_size, fontname=font_name)
    else:
        ax.yaxis.set_visible(False)

    ax.set_xticks(np.arange(len(x_values_list[0])))
    ax.set_xticklabels(x_values_list[0], rotation=45, ha='right')

    if title_option:
        plt.title(data['title'], loc='center', fontsize=font_size + 2, fontname=font_name)

    if legend_option == True:
        ncol = random.choice([1, 2, 3])
        shadow = random.choice([True, False])
        frameon = random.choice([True, False])
        ax.legend(loc=legend_loc_option[0], ncol=ncol, shadow=shadow, frameon=frameon, fancybox=True, framealpha=0.7, bbox_to_anchor=legend_loc_option[1])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    if not title_option:
        data['title'] = ""
    if not axes_option and value_labels_option:
        data['y_axis'] = ""

    with open(json_save_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
