import json
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D
import matplotlib
from matplotlib import font_manager
font_manager._load_fontmanager()

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
    """Plot a horizontal grouped bar chart based on the input JSON with variations."""
    data = extract_data(json_input)

    spine_option = random.choice(['visible', 'light', 'transparent'])

    # Define possible options
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

    # Number of groups and categories
    num_groups = len(data['models'])
    num_categories = len(data['models'][0]['x'])

    categories = data['models'][0]['x']
    y = np.arange(len(categories))
    height = 0.8 / num_groups

    fig, ax = plt.subplots()
    legend_elements = []

    for i, model in enumerate(data['models']):
        x_values = []
        units = []
        for value_str in model['y']:
            value, unit = convert_to_float_and_extract_unit(value_str)
            if value is not None:
                x_values.append(value)
                units.append(unit)
            else:
                print(f"Warning: Conversion failed for value {value_str}")
                return

        x = np.array(x_values, dtype=float)
        if len(x) != len(y):
            print("Error: The number of x values does not match the number of categories.")
            return

        color = model['colors'][0]
        bars = ax.barh(y - height * (num_groups - 1) / 2 + i * height, x, height, label=model['name'], color=color)

        if model['name'] not in [element.get_label() for element in legend_elements]:
            legend_elements.append(Line2D([0], [0], marker=marker_style, color='w', label=model['name'],
                                          markerfacecolor=color, markersize=10, linestyle='None'))

        if values_option:
            for bar, unit in zip(bars, units):
                width = bar.get_width()
                if width.is_integer():
                    text_format = f"{width:.0f}"
                else:
                    text_format = f"{width:.1f}"

                text = unit + text_format if unit.strip() == '$' else text_format + unit if unit.strip() else text_format
                if values_option == 'inside' and width > max(x) * 0.05:
                    ax.text(width - max(x) * 0.03, bar.get_y() + bar.get_height() / 2, text, ha='right', va='center', fontsize=font_size, fontname=font_name, color='white')
                elif values_option == 'outside':
                    ax.text(width + max(x) * 0.01, bar.get_y() + bar.get_height() / 2, text, ha='left', va='center', fontsize=font_size, fontname=font_name)

    # 修正前のy-ticks設定
    # ax.set_yticks(y + height * (num_groups -3) / 2)
    
    # 修正後のy-ticks設定（グループの中央に設定）
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=font_size, fontname=font_name)

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
