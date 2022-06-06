import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import os, json
from utils import load_checkpoint
import seaborn as sns
import matplotlib.ticker as ticker

# ## medical dataset
# dataset = "medical"
# neighbour_len_buf = [10, 10, 10, 10, 10]
# checkpoint_fname_buf_rs = "./log_rs/checkpoint_medical/checkpoint_medical-20210714-115219/test_measures_ts.pth.tar"   # lambda 0.15
#
# checkpoint_fname_buf_us = "./log_us/checkpoint_medical/checkpoint_medical-20210714-105607/test_measures_ts.pth.tar"   # lambda 0.15
#
# checkpoint_fname_buf_csa = "./log_csa/checkpoint_medical/checkpoint_medical-20210714-100133/test_measures_ts.pth.tar"   # lambda 0.15
# # checkpoint_fname_buf_csa = "./log_csa/checkpoint_medical/checkpoint_medical-20210714-112057/test_measures_ts.pth.tar"   # lambda 0.2
# # checkpoint_fname_buf_csa = "./log_csa/checkpoint_medical/checkpoint_medical-20210714-112501/test_measures_ts.pth.tar"   # lambda 0.5
#
# checkpoint_fname_buf_apd = "./log/checkpoint_medical/checkpoint_medical-20210714-111542/test_measures_ts.pth.tar" # lambda 0.15
#
# checkpoint_fname_buf_vln = "./log/checkpoint_medical/checkpoint_medical-20210714-111542/vallina_measures_efficiency.pth.tar"


## Adult dataset
dataset = "Adult"
neighbour_len_buf = [10, 10, 10, 10, 10]

checkpoint_fname_buf_rs = "log_rs/checkpoint_adult/checkpoint_adult-20210714-143928/test_measures_ts.pth.tar"   # lambda 0.5

# checkpoint_fname_buf_us = "./log_us/checkpoint_adult/checkpoint_adult-20210714-143314/test_measures_ts.pth.tar"   # lambda 0.03
checkpoint_fname_buf_us = "./log_us/checkpoint_adult/checkpoint_adult-20210714-125136/test_measures_ts.pth.tar"   # lambda 0.5
# checkpoint_fname_buf_us = "./log_us/checkpoint_adult/checkpoint_adult-20210714-140821/test_measures_ts.pth.tar"   # lambda 2
# checkpoint_fname_buf_us = "./log_us/checkpoint_adult/checkpoint_adult-20210714-140234/test_measures_ts.pth.tar"   # lambda 5

checkpoint_fname_buf_csa = "log_csa/checkpoint_adult/checkpoint_adult-20210714-144847/test_measures_ts.pth.tar"   # lambda 0.5

# checkpoint_fname_buf_apd = "./log/checkpoint_adult/checkpoint_adult-20210625-212415/test_measures_ts.pth.tar" # lambda 0.03
# checkpoint_fname_buf_apd = "./log/checkpoint_adult/checkpoint_adult-20210714-134726/test_measures_ts.pth.tar" # lambda 0.1
checkpoint_fname_buf_apd = "./log/checkpoint_adult/checkpoint_adult-20210714-124328/test_measures_ts.pth.tar" # lambda 0.5
# checkpoint_fname_buf_apd = "./log/checkpoint_adult/checkpoint_adult-20210714-141301/test_measures_ts.pth.tar" # lambda 2

checkpoint_fname_buf_vln = "./log/checkpoint_adult/checkpoint_adult-20210714-124328/vallina_measures_efficiency.pth.tar"


# Default
# dataset = "default"
# neighbour_len_buf = [10, 10, 10, 10, 10]
# checkpoint_fname_buf_rs = "./log_rs/checkpoint_default/checkpoint_default-20210714-164224/test_measures_ts.pth.tar"   # lambda 0.1
#
# checkpoint_fname_buf_us = "./log_us/checkpoint_default/checkpoint_default-20210714-163621/test_measures_ts.pth.tar"   # lambda 0.1
# # checkpoint_fname_buf_us = "./log_us/checkpoint_default/checkpoint_default-20210706-172545/test_measures_ts.pth.tar"   # lambda 0.2
#
# # checkpoint_fname_buf_csa = "./log_csa/checkpoint_default/checkpoint_default-20210706-195715/test_measures_ts.pth.tar" # lambda 0.02
# checkpoint_fname_buf_csa = "./log_csa/checkpoint_default/checkpoint_default-20210706-193025/test_measures_ts.pth.tar" # lambda 0.1
#
# # checkpoint_fname_buf_apd = "./log/checkpoint_default/checkpoint_default-20210706-170754/test_measures_ts.pth.tar" # lambda 0.02
# checkpoint_fname_buf_apd = "./log/checkpoint_default/checkpoint_default-20210706-164429/test_measures_ts.pth.tar" # lambda 0.1
#
# checkpoint_fname_buf_vln = "./log/checkpoint_default/checkpoint_default-20210706-164429/vallina_measures_efficiency.pth.tar"



checkpoint_buf_buf = [checkpoint_fname_buf_rs, checkpoint_fname_buf_us, checkpoint_fname_buf_csa, checkpoint_fname_buf_apd, checkpoint_fname_buf_vln]
marker_buf = ['v', '>', '<', 'o', '^']
legend_buf = ["Random", "Uncertainty", "Core-set", "APD", "Vanilla"]
color_buf = ["blue", "green", "black", "red", "orange"]

def curve_smooth(x, neighbour_len=2):

    print(x.shape)

    # x_mean = x.mean(axis=1)
    x_mean = np.array([x[max(idx - neighbour_len, 0):idx + 1].mean()
                        for idx in range(len(x))])

    # x_mean = np.array([x[idx:min(len(x), idx+neighbour_len)].mean()
    #                    for idx in range(len(x))])

    x_ucb = x.max(axis=1)
    # x_ucb = np.array([x[max(idx - neighbour_len, 0):idx + 1].max()
    #                     for idx in range(len(x))])

    x_lcb = x.min(axis=1)
    # x_lcb = np.array([x[max(idx - neighbour_len, 0):idx + 1].min()
    #                     for idx in range(len(x))])

    return x_mean, x_ucb, x_lcb

def main():

    checkpoint_vln = load_checkpoint(checkpoint_fname_buf_vln)
    eo_vln = np.array([[y['equal_odds'] for y in x] for x in checkpoint_vln["choose_test_measures_buffer"]]).mean(axis=0)
    max_x = 0


    for alg_index, checkpoint_fname in enumerate(checkpoint_buf_buf):

        # x_data = []
        # y_data = []

        if len(checkpoint_fname) == 0:
            continue

        print(checkpoint_fname)

        checkpoint_active = load_checkpoint(checkpoint_fname)
        test_measures_active = checkpoint_active["choose_test_measures_buffer"]
        x_data = np.array(checkpoint_active["x_data"])*2
        y_data = np.array([[y['equal_odds'] for y in x] for x in test_measures_active]).T

        print(x_data)
        print(y_data)
        print(neighbour_len_buf[alg_index])
        y_mean, y_ucb, y_lcb = curve_smooth(y_data, neighbour_len=neighbour_len_buf[alg_index])
        # y_plot = [y_mean, y_ucb, y_lcb]

        marker = marker_buf[alg_index]
        # sns.tsplot(time=x_data, data=y_plot, color=color_buf[alg_index],
        #            condition=legend_buf[alg_index])
        # plt.plot(x_data, y_data.T.mean(axis=0), marker=marker, label=legend_buf[alg_index],
        #      linewidth=4.0, markersize=10, color=color_buf[alg_index])
        plt.plot(x_data[::5], y_mean[::5], marker=marker, label=legend_buf[alg_index],
                    linewidth=4.0, markersize=10, color=color_buf[alg_index])

        plt.plot([0, x_data[0]], [eo_vln, y_mean[0]], # linestyle="--",
                    linewidth=4.0, color=color_buf[alg_index])

        max_x = x_data[-1] if x_data[-1] > max_x else max_x


        #####################



        #####################

    # plt.figure()

    plt.xlabel("Annotated instance ratio", fontsize=15)
    plt.ylabel("Equalized Odds", fontsize=15)
    plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
    # plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-3e-5, max_x+3e-5])
    plt.grid()
    plt.subplots_adjust(left=0.14, bottom=0.11, top=0.97, right=0.99, wspace=0.01)
    plt.legend(loc='lower right', fontsize=15, frameon=True)
    # plt.savefig(os.path.join("figure/", './ACC_vs_EO_' + dataset + '.png'))
    plt.savefig(os.path.join("figure/", './EO_vs_label_' + dataset + '.pdf'))

    plt.show()





if __name__ == "__main__":
    main()

