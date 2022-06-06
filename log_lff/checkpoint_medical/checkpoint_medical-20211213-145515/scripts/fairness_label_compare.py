import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import os, json
from utils import load_checkpoint
import seaborn as sns
import matplotlib.ticker as ticker

# ## CelebA Wavy hair
dataset = "Wavy_hair"
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-163402/test_measures_ts.pth.tar" # lambda 1.2 label 0.03
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-171106/test_measures_ts.pth.tar" # lambda 1.8 label 0.03
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-171716/test_measures_ts.pth.tar" # lambda 1.2 label 0.05
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-213535/test_measures_ts.pth.tar" # lambda 0.5 label 0.03
checkpoint_fname_apd_0_5 = "./log/checkpoint_celebA/checkpoint_celebA-20210716-194527/test_measures_ts.pth.tar" # lambda 0.5 label 0.05
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-214222/test_measures_ts.pth.tar", checkpoint_fname# lambda 0.8 label 0.03
checkpoint_fname_apd_0_8 = "./log/checkpoint_celebA/checkpoint_celebA-20210715-214827/test_measures_ts.pth.tar" # lambda 0.8 label 0.05
# checkpoint_fname_apd = "./log/checkpoint_celebA/checkpoint_celebA-20210715-215653/test_measures_ts.pth.tar" # lambda 0.8 label 0.1


checkpoint_fname_buf_rs_0_8 = [
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190713/test_measures.pth.tar",   # lambda 0.8 label 0.05
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190615/test_measures.pth.tar",   # lambda 0.8 label 0.04
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210720-123517/test_measures.pth.tar",   # lambda 0.8 label 0.038
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190127/test_measures.pth.tar",   # lambda 0.8 label 0.03
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-192659/test_measures.pth.tar", # lambda 0.8 label 0.025
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190346/test_measures.pth.tar", # lambda 0.8 label 0.02
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-192620/test_measures.pth.tar",   # lambda 0.8 label 0.015
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190405/test_measures.pth.tar",   # lambda 0.8 label 0.01
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190813/test_measures.pth.tar",   # lambda 0.8 label 0.005
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-190922/test_measures.pth.tar", # lambda 0.8 label 0.002
]
checkpoint_fname_buf_rs_0_5 = [
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-195610/test_measures.pth.tar", ## lambda 0.5 label 0.05
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-195720/test_measures.pth.tar", ## lambda 0.5 label 0.04
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210720-120648/test_measures.pth.tar", ## lambda 0.5 label 0.038
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210720-120421/test_measures.pth.tar", ## lambda 0.5 label 0.035
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-195759/test_measures.pth.tar", ## lambda 0.5 label 0.03
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210720-120220/test_measures.pth.tar", ## lambda 0.5 label 0.025
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210720-120912/test_measures.pth.tar", ## lambda 0.5 label 0.022
        # "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-195845/test_measures.pth.tar", ## lambda 0.5 label 0.02
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-195952/test_measures.pth.tar", ## lambda 0.5 label 0.01
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-200423/test_measures.pth.tar", ## lambda 0.5 label 0.007
        "./log_rs/checkpoint_celebA/checkpoint_celebA-20210716-200630/test_measures.pth.tar", ## lambda 0.5 label 0.003
]

checkpoint_fname_buf_fullabel = [
        "./log_fulllabel/checkpoint_celebA/checkpoint_celebA-20210715-214225/test_measures.pth.tar", ## lambda 0.5
        "./log_fulllabel/checkpoint_celebA/checkpoint_celebA-20210716-194906/test_measures.pth.tar",  ## lambda 0.8
    # "./log_fulllabel/checkpoint_celebA/checkpoint_celebA-20210720-122014/test_measures.pth.tar", ## lambda 0.05
]

checkpoint_fname_vln = "./log/checkpoint_celebA/checkpoint_celebA-20210716-194527/vallina_measures_efficiency.pth.tar"
checkpoint_vln = load_checkpoint(checkpoint_fname_vln)
eo_vln = np.array([[y['equal_odds'] for y in x] for x in checkpoint_vln["choose_test_measures_buffer"]]).mean(axis=0)

marker_buf = ['v', '>', '<', 'o', '^']
legend_buf = ["Random", "US", "CSA", "APD", "Vanilla"]
color_buf = ["blue", "green", "black", "red", "orange"]

def curve_smooth(x, neighbour_len=2):

    # print(x.shape)

    # x_mean = x.mean(axis=1)
    x_mean = np.array([x[max(idx - neighbour_len, 0):idx + 1].mean()
                        for idx in range(len(x))])

    # x_mean = np.array([x[idx:min(len(x), idx+neighbour_len)].mean()
    #                    for idx in range(len(x))])

    # x_ucb = x.max(axis=1)
    x_ucb = np.array([x[max(idx - neighbour_len, 0):idx + 1].max()
                        for idx in range(len(x))])
    #
    # x_lcb = x.min(axis=1)
    x_lcb = np.array([x[max(idx - neighbour_len, 0):idx + 1].min()
                        for idx in range(len(x))])

    return x_mean, x_ucb, x_lcb


def fullabel_plot(marker, color, legend, linestyle):

    for checkpoint_index, checkpoint_fname in enumerate(checkpoint_fname_buf_fullabel):
        print(checkpoint_fname)

        checkpoint_active = load_checkpoint(checkpoint_fname)
        test_measures_active = checkpoint_active["choose_test_measures_buffer"]
        x_data = checkpoint_active["x_data"]
        # x_data = checkpoint_active["max_label_number"]
        y_data = np.array([[y['equal_odds'] for y in x] for x in test_measures_active]).mean(axis=0)

        print(x_data)
        print(y_data)

        plt.plot([0,5e-2], [y_data, y_data], label=legend[checkpoint_index], marker=marker[checkpoint_index],
                 linewidth=4.0, markersize=10, color=color[checkpoint_index], linestyle=linestyle[checkpoint_index])


def rs_plot(checkpoint_fname_buf, marker, color, legend):
    x_data = []
    y_data = []
    for checkpoint_fname in checkpoint_fname_buf:

        if len(checkpoint_fname) == 0:
            continue

        print(checkpoint_fname)

        checkpoint_active = load_checkpoint(checkpoint_fname)
        test_measures_active = checkpoint_active["choose_test_measures_buffer"]
        x_data_instance = checkpoint_active["x_data"]
        # x_data = checkpoint_active["max_label_number"]
        y_data_instance = np.array([[y['equal_odds'] for y in x] for x in test_measures_active]).mean(axis=0)

        x_data.append(x_data_instance)
        y_data.append(y_data_instance)

    plt.plot(x_data, y_data, label=legend, marker=marker,
         linewidth=4.0, markersize=10, color=color)

    plt.plot([0, x_data[-1][0]], [eo_vln, y_data[-1][0]], marker=marker,
         linewidth=4.0, markersize=10, color=color)


def apd_plot(checkpoint_fname, neighbour_len, marker, color, legend):

    checkpoint_active = load_checkpoint(checkpoint_fname)
    test_measures_active = checkpoint_active["choose_test_measures_buffer"]
    x_data = checkpoint_active["x_data"]
    y_data = np.array([[y['equal_odds'] for y in x] for x in test_measures_active]).mean(axis=0)

    # print(x_data)
    # print(y_data)
    y_mean, y_ucb, y_lcb = curve_smooth(y_data, neighbour_len=neighbour_len)

    plt.plot(x_data[::5], y_mean[::5], label=legend, marker=marker,
             linewidth=4.0, markersize=10, color=color)
    plt.plot([0, x_data[0]], [eo_vln, y_data[0]], marker=marker,
            linewidth=4.0, markersize=10, color=color)

def vln_plot(checkpoint_fname, marker, color, legend):
    checkpoint_vln = load_checkpoint(checkpoint_fname)
    test_measures_vln = checkpoint_vln["choose_test_measures_buffer"]
    x_data = checkpoint_vln["x_data"]
    y_data = np.array([[y['equal_odds'] for y in x] for x in test_measures_vln]).mean(axis=0)

    plt.plot(x_data, y_data, label=legend, marker=marker,
         linewidth=4.0, markersize=10, color=color)


def main():

    # max_x = 0


    apd_plot(checkpoint_fname_apd_0_5, 20, 'o', 'black', "APD $\lambda = 0.5$")
    apd_plot(checkpoint_fname_apd_0_8, 20, 's', 'blue', "APD $\lambda = 0.8$")
    rs_plot(checkpoint_fname_buf_rs_0_5, '^', 'green', "Random $\lambda = 0.5$")
    rs_plot(checkpoint_fname_buf_rs_0_8, 'v', 'c', "Random $\lambda = 0.8$")
    fullabel_plot(['', ''], ['red', 'red'], ["Full labels $\lambda = 0.5$",
        "Full labels $\lambda = 0.8$"], ["--", ":"])
    vln_plot(checkpoint_fname_vln, "^", "orange", "Vanilla")

    plt.xlabel("Labeled instance ratio", fontsize=15)
    plt.ylabel("$\Delta$EO", fontsize=15)
    plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    plt.gca().ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-2e-5, 4e-2 + 2e-5])
    plt.grid()
    plt.subplots_adjust(left=0.14, bottom=0.11, top=0.97, right=0.99, wspace=0.01)
    plt.legend(loc='lower right', fontsize=15, frameon=True)
    plt.savefig(os.path.join("figure/", './EO_vs_label_' + dataset + '.pdf'))

    plt.show()


if __name__ == "__main__":
    main()

