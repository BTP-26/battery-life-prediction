import os
import numpy as np
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sympy.physics.control.control_plots import matplotlib
from tqdm import tqdm
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
font = {'family': 'Arial'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42  # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42


# matplotlib.rc('font', **font0)
def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(ax, fontsize=11.7):
    ax.tick_params(axis='y',
                   labelsize=fontsize  # y轴字体大小设置
                   )
    ax.tick_params(axis='x',
                   labelsize=fontsize  # x轴字体大小设置
                   )
    
def draw_third_b():
    path = './dataset//third_fig_plot_data/'
    label_path = './dataset/'
    files_path = os.listdir(path)
    labels_path = os.listdir(label_path)
    files = [i for i in files_path if i.endswith('.csv')]
    labels = [i for i in labels_path if i.endswith('.json')]
    selected_cells = []
    for label in labels:
        label_data = json.load(open(f'{label_path}{label}'))
        for key, value in label_data.items():
            key = key + '_third_fig_data.csv'
            selected_cells.append(key)
    intersection = list(set(files) & set(selected_cells))
    cycles = []
    total_soh = []
    file_name_list = []
    for file in intersection:# load the cell data
        if file.startswith('Tongji1_CY45-05_1--19'):
            continue
        elif file.startswith('ISU-ILCC_G42C4'):
            continue
        elif file.startswith('NA-ion'):
            if 'DefaultGroup' in file:
                continue
        elif file.startswith('NA-coin'):
            continue

        file_name = file.split('_third_fig_data')[0]
        cell_df = pd.read_csv(path + file)
        soh = cell_df['SOH'].values
        last_soh = soh[-1]
        if last_soh < 0.78:
            continue
        cycle = max(cell_df['Cycle number'].values)
        cycles.append(cycle)
        total_soh.append(soh)
        file_name_list.append(file_name)

    # Plot results
    fig = plt.figure(figsize=(6,3))
    plt.xlabel('Cycle number', fontsize='20')
    plt.ylabel('SOH', fontsize='20')
    plt.grid(alpha=.3)

    cycle_min = min(cycles)
    cycle_max = max(cycles)
    norm = matplotlib.colors.Normalize(vmin=cycle_min, vmax=cycle_max)
    colormap = sns.color_palette("coolwarm_r", as_cmap=True)
    # cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), ax=plt.gca())
    # cb.set_label('Cycle life', fontsize='15')

    for soh, cycle, file_name in zip(total_soh, cycles, file_name_list):
        color = colormap(norm(cycle))
        plt.plot(range(1, cycle + 1), soh, 'k-', c=color, linewidth=1)
        plt.plot(range(1, cycle + 1), soh, 'k-', c=color, linewidth=1)

    set_ax_linewidth(plt.gca())
    set_ax_linewidth(plt.gca())
    # ax.set_xticklabels(fontsize=15)
    fig.tight_layout()
    plt.savefig('./figures/SOH_trajectories.jpg', dpi=600)
    plt.savefig('./figures/SOH_trajectories.pdf')

def draw_third_a():
    path = './dataset/'
    files_path = os.listdir(path)
    json_files = [i for i in files_path if i.endswith('labels.json')]
    cycles_length = []
    type = []
    zn_sample = 0
    na_sample = 0
    li_sample = 0
    calb_sample = 0
    for file in tqdm(json_files):
        
        with open(path + file, 'rb') as f:
            cell = json.load(f)
            for key, value in cell.items():
                if key.startswith('ISU-ILCC_G42C4'):
                    continue
                if 'ZN-coin_labels' in file:
                    zn_sample += 1
                    type.append('Zn-ion')
                    cycles_length.append(value)
                    # continue
                elif 'NA-ion_labels' in file:
                    na_sample += 1
                    type.append('Na-ion')
                    cycles_length.append(value)
                elif 'MICH_EXP_labels' in file:
                    continue
                elif 'MICH_labels' in file:
                    continue
                elif 'CALB' in file:
                    calb_sample += 1
                    type.append('CALB')
                    cycles_length.append(value)
                    # continue
                else:
                    li_sample += 1
                    type.append('Li-ion')
                    cycles_length.append(value)
                    # continue
    data = {
        'cycles_length': cycles_length,
        'Battery Type': type
    }
    df = pd.DataFrame(data)
    fig = plt.figure(figsize=(6,3))
    # sns.histplot(data=df, x="cycles_length", hue="battery_type", multiple="stack", binwidth=100)
    sns.histplot(data=df, x="cycles_length", hue="Battery Type", binwidth=100,
                 hue_order=['Na-ion', 'CALB', 'Zn-ion', 'Li-ion'], palette=['#EA5C49', '#EB9401', '#785FE6', '#AAE9A0'])
    plt.ylabel('Count', fontsize='20')
    plt.xlabel('Life label', fontsize='20')
    set_ax_linewidth(plt.gca())
    fig.tight_layout()
    plt.savefig('./figures/life_distribution.jpg', dpi=600)
    plt.savefig('./figures/life_distribution.pdf')

    # plt.title('Battery Life Histogram', fontsize='15')

# fig = plt.figure(figsize=(6,6))
draw_third_a()
draw_third_b()

# plt.subplots_adjust(wspace =0, hspace =0.35)
# plt.savefig('./figures/third_fig.jpg', dpi=600)
# plt.savefig('./figures/third_fig.pdf')