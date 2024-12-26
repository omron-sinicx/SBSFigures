import json
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.lines import Line2D

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
    """Plot a vertical bar chart based on the input JSON with random configuration."""
    # Extract data from JSON
    data = extract_data(json_input)

    # Skip processing if there are multiple models
    if len(data['models']) > 1:
        print("More than one model found, skipping processing.")
        return

    # Define data
    x = data['models'][0]['x']
    y = []
    units = []

    for y_value in data['models'][0]['y']:
        value, unit = convert_to_float_and_extract_unit(y_value)
        if value is not None:
            y.append(value)
            units.append(unit)
        else:
            print(f"Warning: Conversion failed for value {y_value}")
            return

    y = np.array(y, dtype=float)
    colors = data['models'][0]['colors']

    # Randomly choose options for chart features
    title_option = random.choice([True, False])
    axes_option = random.choice([True, False])
    values_option = random.choice(['inside', 'outside', None]) if axes_option else random.choice(['inside', 'outside'])
    legend_option = random.choice([True, False])
    title_loc_option = random.choice(['left', 'center', 'right'])
    legend_loc_option = random.choice([('upper left', (1, 0.9)), ('lower right', (1, 0)), ('upper right', (1, 1)), ('lower left', (0, 0)), ('upper center', (0.5, 1))])
    spine_option = random.choice(['visible', 'light', 'transparent'])

    fig, ax = plt.subplots()

    # Create vertical bars
    bars = plt.bar(x, y, color=colors)

    # Add title if selected
    if title_option:
        plt.title(data['title'], fontsize=font_size + 2, fontname=font_name, loc=title_loc_option)

    # Set x-axis labels
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)

    # Add axes labels if selected
    if axes_option:
        ax.set_xlabel(data['x_axis'], fontsize=font_size, fontname=font_name)
        ax.set_ylabel(data['y_axis'], fontsize=font_size, fontname=font_name)
    else:
        ax.yaxis.set_visible(False)

    # Create legend if selected
    if legend_option:
        ncol = random.choice([1, 2, 3])
        shadow = random.choice([True, False])
        frameon = random.choice([True, False])
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label=x_value, markerfacecolor=color, markersize=10, linestyle='None')
            for x_value, color in zip(data['models'][0]['x'], data['models'][0]['colors'])
        ]
        ax.legend(handles=legend_elements, loc=legend_loc_option[0], ncol=ncol, shadow=shadow, frameon=frameon, fancybox=True, framealpha=0.7, bbox_to_anchor=legend_loc_option[1])

    # Add numerical values to bars if selected
    if values_option:
        for bar in bars:
            yval = bar.get_height()
            text_format = f"{yval:.0f}" if yval.is_integer() else f"{yval:.1f}"
            text = unit + text_format if unit.strip() == '$' else text_format + unit if unit.strip() else text_format
            if values_option == 'inside' and yval > max(y) * 0.05:
                plt.text(bar.get_x() + bar.get_width()/2, yval - max(y) * 0.03, text, ha='center', va='top', fontsize=font_size, fontname=font_name, color='white')
            elif values_option == 'outside':
                plt.text(bar.get_x() + bar.get_width()/2, yval + max(y) * 0.01, text, ha='center', va='bottom', fontsize=font_size, fontname=font_name)

    # Adjust spine visibility based on option
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

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Update JSON fields if options were disabled
    if not title_option:
        data['title'] = ""
    if not axes_option:
        data['y_axis'] = ""

    # Save updated JSON
    with open(json_save_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
