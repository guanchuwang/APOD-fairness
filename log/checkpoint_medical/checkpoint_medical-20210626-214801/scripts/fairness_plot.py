import matplotlib.pyplot as plt
import torch
# import seaborn as sns
import numpy as np
import os, json
from utils import load_checkpoint
import seaborn as sns
import matplotlib.ticker as ticker

# test_hyperparam_fname1 = "./log/checkpoint_adult/checkpoint_adult-20210515-191828/adult.json"
test_hyperparam_fname1 = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-165448/ucla_law_race.json" # K=20
# test_hyperparam_fname1 = "./log/checkpoint_medical/checkpoint_medical-20210515-153212/medical.json"
# test_hyperparam_fname1 = "./log/checkpoint_bank/checkpoint_bank-20210515-212355/bank.json"
# test_hyperparam_fname1 = "./log/checkpoint_region_job_z/checkpoint_region_job_z-20210512-185156/region_job_z.json"
# test_hyperparam_fname1 = "./log/checkpoint_region_job_n/checkpoint_region_job_n-20210512-161132/region_job_n.json"

# test_hyperparam_fname2 = "./log_rs/checkpoint_adult/checkpoint_adult-20210515-201732/adult.json"
test_hyperparam_fname2 = "./log_rs/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-171540/ucla_law_race.json" # K=20
# test_hyperparam_fname2 = "./log_rs/checkpoint_medical/checkpoint_medical-20210515-154807/medical.json"
# test_hyperparam_fname2 = "./log_rs/checkpoint_bank/checkpoint_bank-20210515-220146/bank.json"
# test_hyperparam_fname2 = "./log_rs/checkpoint_region_job_z/checkpoint_region_job_z-20210512-194411/region_job_z.json"
# test_hyperparam_fname2 = "./log_rs/checkpoint_region_job_n/checkpoint_region_job_n-20210512-165627/region_job_n.json"

with open(test_hyperparam_fname1) as json_file:
    hyperparam1 = json.load(json_file)

with open(test_hyperparam_fname2) as json_file:
    hyperparam2 = json.load(json_file)

def main():

    checkpoint_name = os.path.join(hyperparam1["root_log_dir"], "test_measures.pth.tar")
    checkpoint_active = load_checkpoint(checkpoint_name)
    test_measures_active = checkpoint_active["choose_test_measures_buffer"]

    checkpoint_sss_name = os.path.join(hyperparam2["root_log_dir"], "test_measures.pth.tar")
    checkpoint_sss = load_checkpoint(checkpoint_sss_name)
    test_measures_sss = checkpoint_sss["choose_test_measures_buffer"]

    # mean_accuracy_active = np.array([[y['accuracy'] for y in x] for x in test_measures_active]).mean(axis=0)
    # mean_objective_active = np.array([[y['objective'] for y in x] for x in test_measures_active]).mean(axis=0)
    mean_fpr_diff_active = np.array([[y['fpr_diff'] for y in x] for x in test_measures_active])
    mean_fnr_diff_active = np.array([[y['fnr_diff'] for y in x] for x in test_measures_active])

    # mean_accuracy_sss = np.array([[y['accuracy'] for y in x] for x in test_measures_sss]).mean(axis=0)
    # mean_objective_sss = np.array([[y['objective'] for y in x] for x in test_measures_sss]).mean(axis=0)
    mean_fpr_diff_sss = np.array([[y['fpr_diff'] for y in x] for x in test_measures_sss])
    mean_fnr_diff_sss = np.array([[y['fnr_diff'] for y in x] for x in test_measures_sss])

    # plt.plot(mean_fpr_diff_active, c='blue')
    # plt.plot(mean_fnr_diff_active, c='red')
    # # plt.plot(mean_objective_active, c='black')
    #
    # plt.plot(mean_fpr_diff_sss, c='blue', linestyle='--')
    # plt.plot(mean_fnr_diff_sss, c='red', linestyle='--')
    # # plt.plot(mean_objective_sss, c='black', linestyle='--')

    start_labels = hyperparam1["warmup_labels"]*2+1

    if hyperparam1["checkpoint_name"] == "checkpoint_bank":
        total_labels = hyperparam1["label_num"] - 10
    else:
        total_labels = hyperparam1["label_num"]

    num_instance_plot = total_labels - start_labels + 1

    plt.figure()
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1000))
    sns.tsplot(time=np.arange(start_labels, total_labels+1), data=mean_fpr_diff_active[:, :num_instance_plot], color='blue', condition="PSS") # , condition=legend_buf[dir_index])
    sns.tsplot(time=np.arange(start_labels, total_labels+1), data=mean_fpr_diff_sss[:, :num_instance_plot], color='red', condition="RS") # , condition=legend_buf[dir_index])
    plt.xlabel("Labeled Instance Number", fontsize=20)
    plt.ylabel("△FPR", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(loc='upper right', fontsize=20, frameon=False)
    plt.savefig(os.path.join(hyperparam1["root_log_dir"], './FPR_GAP_PSS_vs_RS.png'))

    plt.figure()
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1000))
    sns.tsplot(time=np.arange(start_labels, total_labels+1), data=mean_fnr_diff_active[:, :num_instance_plot], color='blue', condition="PSS")  # , condition=legend_buf[dir_index])
    sns.tsplot(time=np.arange(start_labels, total_labels+1), data=mean_fnr_diff_sss[:, :num_instance_plot], color='red', condition="RS")  # , condition=legend_buf[dir_index])
    plt.xlabel("Labeled Instance Number", fontsize=20)
    plt.ylabel("△FNR", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(loc='upper right', fontsize=20, frameon=False)
    plt.savefig(os.path.join(hyperparam1["root_log_dir"], './FNR_GAP_PSS_vs_RS.png'))

    plt.show()


if __name__ == "__main__":
    main()
