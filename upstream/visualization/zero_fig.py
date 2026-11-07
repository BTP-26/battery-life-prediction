import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib
import pandas as pd
import numpy as np
import json

font = {'family': 'Arial'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42  # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42

def get_evenly_spaced_values(x, num_values=300):
    if len(x) < num_values:
        raise ValueError("The list has fewer than 300 elements.")
    
    step = len(x) / num_values
    selected_values = [x[int(i * step)] for i in range(num_values)]
    return selected_values

# matplotlib.rc('font', **font0)
def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y',
                   labelsize=fontsize  # y轴字体大小设置
                   )
    ax.tick_params(axis='x',
                   labelsize=fontsize  # x轴字体大小设置
                   )



def draw_CALB_sequence(fig):
    # with open('./dataset/CALB/CALB_0_B184.pkl', 'rb') as f:
    #     MATR_data = pickle.load(f)

    # length = len(MATR_data['cycle_data'])
    # total_current = []
    # total_voltage = []
    # total_time = []
    # for i in tqdm(range(length)):
    #     if i < 1:
    #         continue
    #     cycle_data = MATR_data['cycle_data'][i]
    #     current = cycle_data['current_in_A']
    #     indices = [i for i, x in enumerate(current) if x == 0]
    #     current = [i for i in current if i not in indices]
    #     voltage = cycle_data['voltage_in_V']
    #     voltage = [i for i in voltage if i not in indices]
    #     times = cycle_data['time_in_s']
    #     times = [i for i in times if i not in indices]
    #     new_times = []
    #     plus_time = 0
    #     for time in times:
    #         time = time.split(' ')[1]
    #         h = float(time.split(':')[0])
    #         m = float(time.split(':')[1])
    #         s = float(time.split(':')[2])
    #         seconds = (h * 3600 + m * 60 + s) + 2527 - 20000
    #         new_times.append(seconds)
    #     time = [time + plus_time for time in new_times]
    #     plus_time = max(time)
    #     # print(min(time))

    #     total_time = total_time + time
    #     total_current = total_current + current
    #     total_voltage = total_voltage + voltage
    #     if i == 3:  # draw first 5 cycles
    #         break

    # ax1 = plt.subplot(2, 2, 2)
    # color = sns.color_palette()[0]
    # ax1.plot(total_time, total_voltage, '-', color=color)
    # ax1.set_xlabel('Time(s)',  fontsize=15)
    # ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    # ax1.tick_params('y', colors=color)
    # ax1.set_ylim(1.0, 4.5)

    # ax2 = ax1.twinx()
    # color = sns.color_palette()[3]
    # ax2.set_ylim(-80, 120)
    # ax2.plot(total_time, total_current, '-', color=color)
    # ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    # ax2.tick_params('y', colors=color)
    # set_ax_linewidth(ax1)

    with open('./dataset/CALB_labels.json', 'rb') as f:
        CALB_label = json.load(f)
    
    max_life = max(CALB_label.values())
    min_cell = ['CALB_0_B183.pkl']
    mean_cell = ['CALB_0_B187.pkl']
    max_cell = ['CALB_0_B188.pkl']
    cells = min_cell + mean_cell + max_cell
    #cells = min_cell + mean_cell

    total_voltage = []
    total_time = []
    for cell in cells:
        with open(f'./dataset/CALB/{cell}', 'rb') as f:
            cell_data = pickle.load(f)
        
        length = len(cell_data['cycle_data'])
        # 1st cycle
        cycle_data = cell_data['cycle_data'][1]
        voltage_1 = cycle_data['voltage_in_V']
        time_1 = cycle_data['time_in_s']
        new_times = []
        for time in time_1:
            new_times.append(time)
        time_1 = [i-new_times[0] for i in new_times]

        # 50th cycle
        cycle_data = cell_data['cycle_data'][49]
        voltage_50 = cycle_data['voltage_in_V']
        time_50 = cycle_data['time_in_s']
        new_times = []
        for time in time_50:
            new_times.append(time)
        time_50 = [i-new_times[0] for i in new_times]

        # 100th cycle
        cycle_data = cell_data['cycle_data'][99]
        voltage_100 = cycle_data['voltage_in_V']
        time_100 = cycle_data['time_in_s']
        new_times = []
        for time in time_100:
            new_times.append(time)
        time_100 = [i-new_times[0] for i in new_times]

        ax1 = plt.subplot(2, 1, 2)
        color = sns.color_palette("flare")
        if cell.endswith('B183.pkl'):
            marker = 'o'
            color = 'red'
            lw=1
            alpha=0.75
            label_prefix = 'Low life cell'
        elif cell.endswith('B187.pkl'):
            marker = '^'
            color = 'green'
            lw=1
            label_prefix = 'Middle life cell'
            alpha=0.5
        elif cell.endswith('B188.pkl'):
            marker = 's'
            color = 'black'
            lw=1
            alpha=0.5
            label_prefix = 'High life cell'

        ax1.plot(time_1, voltage_1, '-', color=color, linestyle='-', label=f'{label_prefix} 1st cycle', alpha=alpha, lw=2)
        ax1.plot(time_50, voltage_50, '-', color=color, linestyle='--', label=f'{label_prefix} 50th cycle', alpha=alpha, lw=2)
        ax1.plot(time_100, voltage_100, '-', color=color, linestyle=':', label=f'{label_prefix} 100th cycle', alpha=alpha, lw=2)
        ax1.set_xlabel('Time(s)',  fontsize=15)
        ax1.set_ylabel('Voltage(V)', color='black',  fontsize=15)
        ax1.tick_params('y', colors='black')
        # ax1.legend()
        set_ax_linewidth(ax1)


def draw_Tongji_sequence(fig):
    # with open('./dataset/Tongji/Tongji1_CY25-05_1-#1.pkl', 'rb') as f:
    #         MATR_data = pickle.load(f)

    # length = len(MATR_data['cycle_data'])
    # total_current = []
    # total_voltage = []
    # total_time = []
    # for i in (range(length)):
    #     cycle_data = MATR_data['cycle_data'][i]
    #     current = cycle_data['current_in_A']
    #     voltage = cycle_data['voltage_in_V']
    #     time = cycle_data['time_in_s']

    #     total_time = total_time + time
    #     total_current = total_current + current
    #     total_voltage = total_voltage + voltage
    #     if i == 2:  # draw first 5 cycles
    #         break

    # ax1 = plt.subplot(2, 2, 1)
    # color = sns.color_palette()[0]
    # ax1.plot(total_time, total_voltage, '-', color=color)
    # ax1.set_xlabel('Time(s)',  fontsize=15)
    # ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    # ax1.tick_params('y', colors=color)

    # ax2 = ax1.twinx()
    # color = sns.color_palette()[3]
    # ax2.set_ylim(-5, 7)
    # ax2.plot(total_time, total_current, '-', color=color)
    # ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    # ax2.tick_params('y', colors=color)
    # set_ax_linewidth(ax1)
    # plt.title('Voltage-Current vs time Profile')
    
    min_cell = ['Tongji1_CY45-05_1-#17.pkl']
    mean_cell = ['Tongji1_CY45-05_1-#18.pkl']
    max_cell = ['Tongji1_CY45-05_1-#12.pkl']
    cells = min_cell + mean_cell + max_cell
    #cells = min_cell + mean_cell

    total_voltage = []
    total_time = []
    for cell_index, cell in enumerate(cells):
        with open(f'./dataset/Tongji/{cell}', 'rb') as f:
            cell_data = pickle.load(f)
        
        length = len(cell_data['cycle_data'])
        # 1st cycle
        cycle_data = cell_data['cycle_data'][0]
        voltage_1 = cycle_data['voltage_in_V']
        time_1 = cycle_data['time_in_s']
        time_1 = [(i-time_1[0]) for i in time_1]
        time_1 = get_evenly_spaced_values(time_1)
        voltage_1 = get_evenly_spaced_values(voltage_1)

        # 50th cycle
        cycle_data = cell_data['cycle_data'][49]
        voltage_50 = cycle_data['voltage_in_V']
        time_50 = cycle_data['time_in_s']
        time_50 = [(i-time_50[0]) for i in time_50]
        time_50 = get_evenly_spaced_values(time_50)
        voltage_50 = get_evenly_spaced_values(voltage_50)

        # 100th cycle
        cycle_data = cell_data['cycle_data'][99]
        voltage_100 = cycle_data['voltage_in_V']
        time_100 = cycle_data['time_in_s']
        time_100 = [(i-time_100[0]) for i in time_100]
        time_100 = get_evenly_spaced_values(time_100)
        voltage_100 = get_evenly_spaced_values(voltage_100)

        ax1 = plt.subplot(2, 1, 1)
        color_palette = sns.color_palette("ch:s=.25,rot=-.25")
        if cell.endswith('CY45-05_1-#17.pkl'):
            marker = 'o'
            color = 'red'
            alpha=0.75
            label_prefix = 'Low life cell'
        elif cell.endswith('CY45-05_1-#18.pkl'):
            marker = '^'
            color = 'green'
            alpha=0.5
            label_prefix = 'Middle life cell'
        elif cell.endswith('CY45-05_1-#12.pkl'):
            marker = 's'
            color = 'black'
            alpha=0.5
            label_prefix = 'High life cell'
        
        ax1.plot(time_1, voltage_1, '-', color=color, linestyle='-', label=f'{label_prefix} 1st cycle', alpha=alpha, lw=2)
        ax1.plot(time_50, voltage_50, '-', color=color, linestyle='--', label=f'{label_prefix} 50th cycle', alpha=alpha, lw=2)
        ax1.plot(time_100, voltage_100, '-', color=color, linestyle=':', label=f'{label_prefix} 100th cycle', alpha=alpha, lw=2)
        ax1.set_xlabel('Time (s)',  fontsize=15)
        ax1.set_ylabel('Voltage (V)', color='black',  fontsize=15)
        ax1.tick_params('y', colors='black')
        # ax1.legend()
        set_ax_linewidth(ax1)


def draw_Tongji_sequence2(fig):
    # with open('./dataset/Tongji/Tongji1_CY25-05_1-#1.pkl', 'rb') as f:
    #         MATR_data = pickle.load(f)

    # length = len(MATR_data['cycle_data'])
    # total_current = []
    # total_voltage = []
    # total_time = []
    # for i in (range(length)):
    #     cycle_data = MATR_data['cycle_data'][i]
    #     current = cycle_data['current_in_A']
    #     voltage = cycle_data['voltage_in_V']
    #     time = cycle_data['time_in_s']

    #     total_time = total_time + time
    #     total_current = total_current + current
    #     total_voltage = total_voltage + voltage
    #     if i == 2:  # draw first 5 cycles
    #         break

    # ax1 = plt.subplot(2, 2, 1)
    # color = sns.color_palette()[0]
    # ax1.plot(total_time, total_voltage, '-', color=color)
    # ax1.set_xlabel('Time(s)',  fontsize=15)
    # ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    # ax1.tick_params('y', colors=color)

    # ax2 = ax1.twinx()
    # color = sns.color_palette()[3]
    # ax2.set_ylim(-5, 7)
    # ax2.plot(total_time, total_current, '-', color=color)
    # ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    # ax2.tick_params('y', colors=color)
    # set_ax_linewidth(ax1)
    # plt.title('Voltage-Current vs time Profile')
    
    min_cell = ['Tongji1_CY45-05_1-#17.pkl']
    mean_cell = ['Tongji1_CY45-05_1-#18.pkl']
    max_cell = ['Tongji1_CY45-05_1-#12.pkl']
    cells = min_cell + mean_cell + max_cell
    #cells = min_cell + mean_cell

    total_voltage = []
    total_time = []
    for cell_index, cell in enumerate(cells):
        with open(f'./dataset/Tongji/{cell}', 'rb') as f:
            cell_data = pickle.load(f)
        
        length = len(cell_data['cycle_data'])
        # 1st cycle
        cycle_data = cell_data['cycle_data'][0]
        voltage_1 = cycle_data['voltage_in_V']
        current_1 = cycle_data['current_in_A']
        time_1 = cycle_data['time_in_s']
        time_1 = [(i-time_1[0]) for i in time_1]
        # time_1 = get_evenly_spaced_values(time_1)
        # voltage_1 = get_evenly_spaced_values(voltage_1)

        # 50th cycle
        cycle_data = cell_data['cycle_data'][49]
        voltage_50 = cycle_data['voltage_in_V']
        current_50 = cycle_data['current_in_A']
        time_50 = cycle_data['time_in_s']
        time_50 = [(i-time_50[0]) for i in time_50]
        # time_50 = get_evenly_spaced_values(time_50)
        # voltage_50 = get_evenly_spaced_values(voltage_50)

        # 100th cycle
        cycle_data = cell_data['cycle_data'][99]
        voltage_100 = cycle_data['voltage_in_V']
        current_100 = cycle_data['current_in_A']
        time_100 = cycle_data['time_in_s']
        time_100 = [(i-time_100[0]) for i in time_100]
        # time_100 = get_evenly_spaced_values(time_100)
        # voltage_100 = get_evenly_spaced_values(voltage_100)
        # current_100 = get_evenly_spaced_values(current_100)

        ax1 = plt.subplot(2, 1, 2)
        color_palette = sns.color_palette("ch:s=.25,rot=-.25")
        if cell.endswith('CY45-05_1-#17.pkl'):
            marker = 'o'
            color = 'red'
            alpha=0.75
            label_prefix = 'Low life cell'
        elif cell.endswith('CY45-05_1-#18.pkl'):
            marker = '^'
            color = 'green'
            alpha=0.5
            label_prefix = 'Middle life cell'
        elif cell.endswith('CY45-05_1-#12.pkl'):
            marker = 's'
            color = 'black'
            alpha=0.5
            label_prefix = 'High life cell'
        
        ax1.plot(time_1, current_1, '-', color=color, linestyle='-', label=f'{label_prefix} 1st cycle', alpha=alpha, lw=2)
        ax1.plot(time_50, current_50, '-', color=color, linestyle='--', label=f'{label_prefix} 50th cycle', alpha=alpha, lw=2)
        ax1.plot(time_100, current_100, '-', color=color, linestyle=':', label=f'{label_prefix} 100th cycle', alpha=alpha, lw=2)
        ax1.set_xlabel('Time (s)',  fontsize=15)
        ax1.set_ylabel('Current (A)', color='black',  fontsize=15)
        ax1.tick_params('y', colors='black')
        # ax1.legend()
        set_ax_linewidth(ax1)

fig = plt.figure(figsize=(6, 5))  # set the size of the figure
draw_Tongji_sequence(fig)
draw_Tongji_sequence2(fig)
fig.tight_layout()
# plt.show()
plt.savefig('./figures/zero_fig.jpg', dpi=600)
plt.savefig('./figures/zero_fig.pdf')
