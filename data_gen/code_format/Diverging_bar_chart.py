import json
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
import re

def extract_data(json_input):
    """Extract data from the JSON input."""
    return json.loads(json_input)

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
    """Plot a horizontal diverging bar chart based on the input JSON with variations."""
    data = extract_data(json_input)

    if len(data['models']) != 2:
        print("Not two models, skipping processing.")
        return

    if data['models'][0]['x'] != data['models'][1]['x']:
        print("Not matching two models length, skipping processing.")
        return

    # Define possible options
    title_option = random.choice([True, False])
    axes_option = random.choice([True, False])
    legend_option = True

    if not axes_option:
        values_option = random.choice(['inside', 'outside'])
    else:
        values_option = random.choice(['inside', 'outside', None])

    title_loc_option = random.choice(['center', 'left', 'right'])
    legend_loc_option = random.choice([('upper left', (1, 0.9)), ('lower right', (1, 0)), 
                                       ('upper right', (1, 1)), ('lower left', (0, 0)), 
                                       ('center', (0.5, 0.5)), ('upper center', (0.5, 1))])
    marker_style = random.choice(['o', 's', 'D', 'v', '<', '>', 'p', 'h', 'H'])
    spine_option = random.choice(['visible', 'light', 'transparent'])

    fig, ax = plt.subplots()

    categories = data['models'][0]['x']
    y_positions = np.arange(len(categories))

    for model_index, model in enumerate(data['models']):
        try:
            y_values = []
            units = []

            for y in model['y']:
                value, unit = convert_to_float_and_extract_unit(y)
                if value is not None:
                    y_values.append(value)
                    units.append(unit)

            if model == data['models'][0]:
                y_values = [-value for value in y_values]

            if len(y_values) != len(y_positions):
                print("Error: Mismatch in lengths of y_values and y_positions.")
                return
        except Exception as e:
            print(f"Error in model '{model['name']}' at index {model_index}: {e}")
            continue

        bars = ax.barh(y_positions, y_values, color=model['colors'][0], label=model['name'])

        if not axes_option:
            for bar, unit in zip(bars, units):
                width = bar.get_width()
                if values_option == 'inside':
                    position = width / 2 if width >= 0 else width / 2
                    color = 'white'
                elif values_option == 'outside':
                    position = width if width >= 0 else width
                    color = 'black'

                if abs(width).is_integer():
                    text = f"{abs(int(width))}{unit}" if unit.strip() else f"{abs(int(width))}"
                else:
                    text = f"{abs(width):.1f}{unit}" if unit.strip() else f"{abs(width):.1f}"

                ax.text(position, bar.get_y() + bar.get_height() / 2, text, ha='center', va='center', fontsize=font_size, color=color)

    ax.set_ylabel(data['x_axis'], fontsize=font_size, fontname=font_name)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)

    if axes_option:
        ax.set_xlabel(data['y_axis'], fontsize=font_size, fontname=font_name)
    else:
        ax.xaxis.set_visible(False)

    if title_option:
        ax.set_title(data['title'], fontsize=font_size + 2, fontname=font_name, loc=title_loc_option)

    if legend_option:
        ncol = random.choice([1, 2, 3])
        shadow = random.choice([True, False])
        frameon = random.choice([True, False])
        legend_elements = []
        for model in data['models']:
            legend_elements.append(Line2D([0], [0], color='w', label=model['name'],
                                        markerfacecolor=model['colors'][0], markersize=10, marker=marker_style))

        ax.legend(handles=legend_elements, loc=legend_loc_option[0], ncol=ncol, shadow=shadow, frameon=frameon, fancybox=True, framealpha=0.7, bbox_to_anchor=legend_loc_option[1])

    if spine_option == 'light':
        for spine in ax.spines:
            ax.spines[spine].set_color('lightgrey')
            ax.spines[spine].set_linewidth(0.5)
    elif spine_option == 'transparent':
        for spine in ax.spines:
            ax.spines[spine].set_color('none')
    else:
        for spine in ax.spines:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    if not title_option:
        data['title'] = ""
    if not axes_option:
        data['y_axis'] = ""

    with open(json_save_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
