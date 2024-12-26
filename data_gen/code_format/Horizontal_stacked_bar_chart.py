import json
import numpy as np
import matplotlib.pyplot as plt
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
    """Plot a horizontal stacked bar chart based on the input JSON with variations."""
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
    y = np.arange(len(categories))

    left_values = np.zeros(len(categories))

    for model_index, model in enumerate(data['models']):
        x_values = []
        units = []
        try:
            for value_str in model['y']:
                value, unit = convert_to_float_and_extract_unit(value_str)
                if value is not None:
                    x_values.append(value)
                    units.append(unit)
                else:
                    print(f"Warning: Conversion failed for value {value_str}")
                    return

            colors = model['colors']
            bars = ax.barh(y, x_values, label=model['name'], color=colors, left=left_values)

            for bar_index, bar in enumerate(bars):
                width = bar.get_width()
                if width.is_integer():
                    text_format = f"{width:.0f}"
                else:
                    text_format = f"{width:.1f}"

                text = text_format + units[bar_index] if units[bar_index].strip() else text_format
                bar_center = bar.get_y() + bar.get_height() / 2
                if values_option == 'inside':
                    text_x_position = left_values[bar_index] + width / 2
                    ax.text(text_x_position, bar_center, text, ha='center', va='center', color='white', fontsize=font_size)
                elif values_option == 'outside':
                    text_x_position = left_values[bar_index] + width
                    ax.text(text_x_position, bar_center, text, ha='left', va='center', color='black', fontsize=font_size)

            left_values += x_values
        except Exception as e:
            print(f"Error in model '{model['name']}' at index {model_index}: {e}")
            continue

    ax.set_ylabel(data['x_axis'], fontsize=font_size, fontname=font_name)
    ax.set_yticks(y)
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
