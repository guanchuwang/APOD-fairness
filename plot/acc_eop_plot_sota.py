import matplotlib.pyplot as plt
import torch
# import seaborn as sns
import numpy as np
import os, json

import sys
sys.path.append("../")

from utils import load_checkpoint
import seaborn as sns
import matplotlib.ticker as ticker
import seaborn as sns


## medical dataset
dataset = "medical"
yaxis_decimal_num = 1
# ant_percent = "8‰"
checkpoint_fname_buf_vln = ["../log/checkpoint_medical/checkpoint_medical-20210624-114757/vallina_measures.pth.tar"]

checkpoint_fname_buf_apd = ["../log/checkpoint_medical/checkpoint_medical-20210624-114757/test_measures.pth.tar", # apd lambda 0.001
                            "../log/checkpoint_medical/checkpoint_medical-20210624-115807/test_measures.pth.tar", # apd lambda 0.005
                            "../log/checkpoint_medical/checkpoint_medical-20210624-115243/test_measures.pth.tar", # apd lambda 0.1
                            "../log/checkpoint_medical/checkpoint_medical-20210626-214801/test_measures.pth.tar", # apd lambda 0.15
                            ]

checkpoint_fname_buf_DRO = [
    "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-173918/test_measures.pth.tar",
    "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-174151/test_measures.pth.tar",
    "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-174930/test_measures.pth.tar",
    "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-181629/test_measures.pth.tar",
    "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-182754/test_measures.pth.tar",
]
checkpoint_fname_buf_lff = [
                           "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145515/test_measures.pth.tar",
                           "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145354/test_measures.pth.tar",
                           "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145849/test_measures.pth.tar",
                           "../log_lff/checkpoint_medical/checkpoint_medical-20211213-150351/test_measures.pth.tar",
                           ]

checkpoint_fname_buf_fal = [
                            "../log_fal/checkpoint_medical/checkpoint_medical-20211212-212056/test_measures.pth.tar", # alpha 0.9 active_buf_size 2
                            "../log_fal/checkpoint_medical/checkpoint_medical-20211212-212353/test_measures.pth.tar", # alpha 0.8 active_buf_size 2
                            "../log_fal/checkpoint_medical/checkpoint_medical-20211212-144022/test_measures.pth.tar", # alpha 0.6  seed 0 active_buf_size 100
                            ]



checkpoint_buf_buf = [checkpoint_fname_buf_vln, checkpoint_fname_buf_DRO, checkpoint_fname_buf_lff, checkpoint_fname_buf_fal, checkpoint_fname_buf_apd]
marker_buf = ['o', 'v', '>', '<', '^', 'o']
legend_buf = ["○ Vanilla", "● Group DRO", "○ LFF", "◔ FAL", "◔ APD"]
color_buf = ["orange", "blue", "green", "darkgoldenrod", "red"]


def main():


    for alg_index, checkpoint_fname_buf in enumerate(checkpoint_buf_buf):

        x_data = []
        y_data = []
        x_data_std = []
        y_data_std = []
        for chackpoint_fname in checkpoint_fname_buf:

            checkpoint_active = load_checkpoint(chackpoint_fname)
            test_measures_active = checkpoint_active["choose_test_measures_buffer"]

            # print(test_measures_active)
            accuracy = np.array([[y['accuracy'] for y in x] for x in test_measures_active])
            accuracy_std = accuracy.std(axis=0)
            mean_accuracy = accuracy.mean(axis=0)
            # mean_equal_odds = -np.array([[y['fnr_diff'] for y in x] for x in test_measures_active]).mean(axis=0)

            tpr0 = np.array([[[1-y['0_fnr']] for y in x] for x in test_measures_active]) + 1e-10
            tpr1 = np.array([[[1-y['1_fnr']] for y in x] for x in test_measures_active]) + 1e-10

            mean_tpr0 = tpr0.mean(axis=0) # np.array([[[1-y['0_fnr']] for y in x] for x in test_measures_active]).mean(axis=0)
            mean_tpr1 = tpr1.mean(axis=0) # np.array([[[1-y['1_fnr']] for y in x] for x in test_measures_active]).mean(axis=0)

            min_tpr = np.concatenate((mean_tpr0, mean_tpr1), axis=1).min(axis=1)
            max_tpr = np.concatenate((mean_tpr0, mean_tpr1), axis=1).max(axis=1)
            mean_ep = min_tpr/max_tpr

            epo = tpr0/tpr1
            epo[epo > 1] = (epo[epo > 1])**(-1)
            epo_std = epo.std(axis=0)

            # mean_fnr0 = np.array([x[0]['0_fnr'] for x in test_measures_active]).mean(axis=0)
            # mean_fnr1 = np.array([x[0]['1_fnr'] for x in test_measures_active]).mean(axis=0)
            # mean_ep = min(1 - mean_fnr0, 1 - mean_fnr1) / max(1 - mean_fnr0, 1 - mean_fnr1)

            # print(mean_tpr0)
            # print(mean_tpr1)


            # print(accuracy_std)
            # print(epo_std)
            x_data.append(mean_ep[-1])
            y_data.append(mean_accuracy[-1])
            x_data_std.append(epo_std[-1])
            y_data_std.append(accuracy_std[-1])

        print(legend_buf[alg_index])
        print(x_data)
        print(y_data)
        print(x_data_std)
        print(y_data_std)
        #

        marker = marker_buf[alg_index]
        plt.plot(x_data, y_data, marker=marker, label=legend_buf[alg_index],
                 linewidth=4.0, markersize=10, color=color_buf[alg_index])

    # plt.ylim([0.82, 0.86])
    # plt.figure()
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1000))
    plt.xlabel("Equality of Opportunity", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=yaxis_decimal_num))
    # plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.subplots_adjust(left=0.13+0.03*yaxis_decimal_num, bottom=0.11, top=0.99, right=0.99, wspace=0.01)
    plt.legend(loc='upper right', fontsize=15, frameon=True)

    plt.show()


if __name__ == "__main__":
    main()


