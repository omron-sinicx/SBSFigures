import json
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
import traceback

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
    """Plot a vertical stacked bar chart based on the input JSON with variations."""
    data = extract_data(json_input)

    title_option = random.choice([True, False])
    axes_option = random.choice([True, False])
    if not axes_option:
        values_option = random.choice(['inside', 'outside'])
    else:
        values_option = random.choice(['inside', 'outside', None])

    legend_option = True
    title_loc_option = random.choice(['left', 'center', 'right'])
    legend_loc_option = random.choice([('upper left', (1, 0.9)), ('lower right', (1, 0)), ('upper right', (1, 1)), ('lower left', (0, 0)), ('upper center', (0.5, 1))])
    spine_option = random.choice(['visible', 'light', 'transparent'])

    fig, ax = plt.subplots()

    categories = data['models'][0]['x']
    x = np.arange(len(categories))

    bottom_values = np.zeros(len(categories))

    for model_index, model in enumerate(data['models']):
        try:
            y_values = []
            units = []
            for value_str in model['y']:
                value, unit = convert_to_float_and_extract_unit(value_str)
                if value is not None:
                    y_values.append(value)
                    units.append(unit)
                else:
                    print(f"Warning: Conversion failed for value {value_str}")
                    return

            colors = model['colors']
            bars = ax.bar(x, y_values, label=model['name'], color=colors, bottom=bottom_values)

            for bar_index, bar in enumerate(bars):
                height = bar.get_height()
                if height.is_integer():
                    text_format = f"{height:.0f}"
                else:
                    text_format = f"{height:.1f}"

                text = text_format + units[bar_index] if units[bar_index].strip() else text_format
                bar_center = bar.get_x() + bar.get_width() / 2
                if values_option == 'inside':
                    text_y_position = bottom_values[bar_index] + height / 2
                    ax.text(bar_center, text_y_position, text, ha='center', va='center', color='white', fontsize=font_size)
                elif values_option == 'outside':
                    text_y_position = bottom_values[bar_index] + height
                    ax.text(bar_center, text_y_position, text, ha='center', va='bottom', color='black', fontsize=font_size)

            bottom_values += y_values
        except Exception as e:
            print(f"Error in model '{model['name']}' at index {model_index}: {e}")
            continue

    ax.set_xlabel(data['x_axis'], fontsize=font_size, fontname=font_name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    if axes_option:
        ax.set_ylabel(data['y_axis'], fontsize=font_size, fontname=font_name)
    else:
        ax.yaxis.set_visible(False)

    if title_option:
        ax.set_title(data['title'], fontsize=font_size + 2, fontname=font_name, loc=title_loc_option)

    if legend_option:
        ncol = random.choice([1, 2, 3])
        shadow = random.choice([True, False])
        frameon = random.choice([True, False])
        marker = random.choice(['o', 's', 'D', 'v', '<', '>', 'p', 'h', 'H'])
        legend_elements = [
            Line2D([0], [0],
                   marker=marker,
                   color='w',
                   label=model['name'],
                   markerfacecolor=model['colors'][0],
                   markersize=10,
                   linestyle='None') for model in data['models']
        ]
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

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    if not title_option:
        data['title'] = ""
    if not axes_option:
        data['y_axis'] = ""

    with open(json_save_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
