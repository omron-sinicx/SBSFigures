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
    """Plot a pie chart based on the input JSON with variations."""
    data = extract_data(json_input)

    labels = [model['name'] for model in data['models']]
    converted_values = []
    units = []

    for model in data['models']:
        converted, unit = convert_to_float_and_extract_unit(str(model['value']))
        if converted is not None:
            converted_values.append(converted)
            units.append(unit)
        else:
            print(f"Warning: Conversion failed for value {model['value']}")
            return

    sizes = np.array(converted_values, dtype=float)
    colors = [model['colors'] for model in data['models']]

    axes_option = random.choice([True, False])
    title_option = random.choice([True, False])
    legend_loc_option = random.choice([('upper left', (1, 0.9)), ('lower right', (1, 0)), ('upper right', (1, 1)), ('lower left', (0, 0)), ('upper center', (0.5, 1))])
    legend_option = random.choice(['outside', 'perimeter', 'lines'])
    title_loc_option = random.choice(['left', 'center', 'right'])

    def custom_autopct(pct):
        total = sum(sizes)
        value = int(round(pct * total / 100.0))
        if value in sizes:
            return '{:.0f}%'.format(pct)
        else:
            return '{:.1f}%'.format(pct)

    if legend_option == 'lines':
        autopct_option = None
    else:
        autopct_option = custom_autopct

    fig, ax = plt.subplots()
    pie_results = ax.pie(
        sizes,
        labels=labels if legend_option in ['perimeter', 'outside'] else None,
        autopct=autopct_option,
        startangle=90,
        colors=colors
    )

    if len(pie_results) == 3:
        wedges, texts, autotexts = pie_results
    else:
        wedges, texts = pie_results
        autotexts = None

    if legend_option == 'inside':
        plt.setp(autotexts, size=font_size, weight="bold", color="white")
    elif legend_option == 'outside':
        ax.legend(wedges, labels, loc=legend_loc_option[0], bbox_to_anchor=legend_loc_option[1], fontsize=font_size)
    elif legend_option == 'perimeter':
        plt.setp(texts, size=font_size, weight="bold")
    elif legend_option == 'lines':
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"

            if sizes[i].is_integer():
                label_format = f'{labels[i]} {sizes[i]:.0f}%'
            else:
                label_format = f'{labels[i]} {sizes[i]:.1f}%'

            ax.annotate(
                label_format,
                xy=(x, y),
                xytext=(1.5*x, 1.2*y),
                horizontalalignment=horizontalalignment,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                fontsize=font_size
            )

    ax.axis('equal')

    if title_option:
        plt.title(data['title'], loc=title_loc_option, fontsize=font_size + 2, fontname=font_name)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    if not title_option:
        data['title'] = ""

    with open(json_save_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
