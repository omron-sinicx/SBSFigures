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
    """Plot a grouped vertical bar chart based on the input JSON with variations."""
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
    marker_style = random.choice(['o', 's', 'D', 'v', '<', '>', 'p', 'h', 'H'])
    spine_option = random.choice(['visible', 'light', 'transparent'])

    num_groups = len(data['models'])

    categories = data['models'][0]['x']
    x = np.arange(len(categories))
    width = 0.8 / num_groups

    fig, ax = plt.subplots()
    legend_elements = []

    for i, model in enumerate(data['models']):
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

        y = np.array(y_values, dtype=float)
        if len(x) != len(y):
            print("Error: The number of x values does not match the number of categories.")
            return

        color = model['colors'][0]
        bars = ax.bar(x + i * width, y, width, label=model['name'], color=color)

        if model['name'] not in legend_elements:
            legend_elements.append(Line2D([0], [0], marker=marker_style, color='w', label=model['name'],
                                          markerfacecolor=color, markersize=10, linestyle='None'))

        if values_option:
            for bar in bars:
                height = bar.get_height()
                if height.is_integer():
                    text_format = f"{height:.0f}"
                else:
                    text_format = f"{height:.1f}"

                text = text_format + unit if unit.strip() else text_format
                if values_option == 'inside' and height > max(y) * 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, height - max(y) * 0.03, text, ha='center', va='top', fontsize=font_size, fontname=font_name, color='white')
                elif values_option == 'outside':
                    ax.text(bar.get_x() + bar.get_width() / 2, height + max(y) * 0.01, text, ha='center', va='bottom', fontsize=font_size, fontname=font_name)

    ax.set_xlabel(data['x_axis'], fontsize=font_size, fontname=font_name)
    ax.set_xticks(x + width / num_groups)
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
