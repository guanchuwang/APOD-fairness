import matplotlib.pyplot as plt
import torch
# import seaborn as sns
import numpy as np
import os, json
from utils import load_checkpoint
import seaborn as sns
import matplotlib.ticker as ticker

# ## medical dataset
checkpoint_fname_buf_vln = ["./log/checkpoint_medical/checkpoint_medical-20210624-114757/vallina_measures.pth.tar"]

checkpoint_fname_buf_apd = ["./log/checkpoint_medical/checkpoint_medical-20210624-114757/test_measures.pth.tar", # apd lambda 0.001
                            "./log/checkpoint_medical/checkpoint_medical-20210624-115807/test_measures.pth.tar", # apd lambda 0.005
                            "./log/checkpoint_medical/checkpoint_medical-20210624-114335/test_measures.pth.tar", # best apd lambda 0.01
                            "./log/checkpoint_medical/checkpoint_medical-20210624-115243/test_measures.pth.tar", # apd lambda 0.1
                            "./log/checkpoint_medical/checkpoint_medical-20210626-214032/test_measures.pth.tar", # apd lambda 0.2
                            # "./log/checkpoint_medical/checkpoint_medical-20210626-213451/test_measures.pth.tar", # apd lambda 0.5
                            ]

checkpoint_fname_buf_rs = [
                           "./log/checkpoint_medical/checkpoint_medical-20210624-114757/rs_fairness/rs_apd_test_measures.pth.tar",   # apd lambda 0.001    
                           "./log/checkpoint_medical/checkpoint_medical-20210624-115807/rs_fairness/rs_apd_test_measures.pth.tar",   # apd lambda 0.005    
                           "./log/checkpoint_medical/checkpoint_medical-20210624-114335/rs_fairness/rs_apd_test_measures.pth.tar",   # best apd lambda 0.01
                           "./log/checkpoint_medical/checkpoint_medical-20210624-115243/rs_fairness/rs_apd_test_measures.pth.tar",   # apd lambda 0.1      
                           ]


# ## credit dataset
# checkpoint_fname_buf_vln = ["./log/checkpoint_bank/checkpoint_bank-20210624-151542/vallina_measures.pth.tar"]
#
# checkpoint_fname_buf_apd = [
#                             "./log/checkpoint_bank/checkpoint_bank-20210624-151542/test_measures.pth.tar", # lambda 0.1
#                             "./log/checkpoint_bank/checkpoint_bank-20210624-151659/test_measures.pth.tar", # lambda 0.5
#                             "./log/checkpoint_bank/checkpoint_bank-20210626-212445/test_measures.pth.tar", # lambda 0.5
#                             # "./log/checkpoint_bank/checkpoint_bank-20210626-212126/test_measures.pth.tar", # lambda 0.7
#                             "./log/checkpoint_bank/checkpoint_bank-20210624-151336/test_measures.pth.tar", # lambda 1.
#                             ]
#
# checkpoint_fname_buf_rs = [
#                            "./log/checkpoint_bank/checkpoint_bank-20210626-211400/rs_fairness/rs_apd_test_measures.pth.tar",  # lambda 0.01
#                            "./log/checkpoint_bank/checkpoint_bank-20210624-151542/rs_fairness/rs_apd_test_measures.pth.tar",  # lambda 0.1
#                            "./log/checkpoint_bank/checkpoint_bank-20210624-151659/rs_fairness/rs_apd_test_measures.pth.tar",  # lambda 0.5
#                            "./log/checkpoint_bank/checkpoint_bank-20210624-151336/rs_fairness/rs_apd_test_measures.pth.tar",  # lambda 1.
#                            ]

# ## ucla_law dataset
# checkpoint_fname_buf_vln = ["./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/vallina_measures.pth.tar"]
#
# checkpoint_fname_buf_apd = ["./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/test_measures.pth.tar",
#                             "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-161439/test_measures.pth.tar",
#                             "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162952/test_measures.pth.tar",
#                             ]
#
# checkpoint_fname_buf_rs = [
#                            # "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-210855/rs_fairness/rs_apd_test_measures.pth.tar",
#                            "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/rs_fairness/rs_apd_test_measures.pth.tar",
#                            "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-161439/rs_fairness/rs_apd_test_measures.pth.tar",
#                            # "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162952/rs_fairness/rs_apd_test_measures.pth.tar",
#                            ]

# ## adult dataset
# checkpoint_fname_buf_vln = [
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-164148/vallina_measures.pth.tar",
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-191308/vallina_measures.pth.tar",  # lambda 0.05
#                             ]
#
# checkpoint_fname_buf_apd = [
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-164148/test_measures.pth.tar", # lambda 0.01
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-204857/test_measures.pth.tar", # lambda 0.03
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-165505/test_measures.pth.tar", # lambda 0.05
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-204118/test_measures.pth.tar", # lambda 0.08
#                             # "./log/checkpoint_adult/checkpoint_adult-20210624-164810/test_measures.pth.tar", # lambda 0.1
#
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-192429/test_measures.pth.tar", # lambda 0.01
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-211903/test_measures.pth.tar", # lambda 0.02
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-212415/test_measures.pth.tar", # lambda 0.03
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-213055/test_measures.pth.tar", # lambda 0.04
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-191308/test_measures.pth.tar", # lambda 0.05
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-194213/test_measures.pth.tar", # lambda 0.1
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-214528/test_measures.pth.tar", # lambda 0.3
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-220012/test_measures.pth.tar", # lambda 0.35
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-215145/test_measures.pth.tar", # lambda 0.4
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-211303/test_measures.pth.tar", # lambda 0.5
#
#                             ]
#
# checkpoint_fname_buf_rs = [
#                             # "./log/checkpoint_adult/checkpoint_adult-20210625-192429/rs_fairness/rs_apd_test_measures.pth.tar", # lambda 0.01
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-211903/rs_fairness/rs_apd_test_measures.pth.tar", # lambda 0.02
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-212415/rs_fairness/rs_apd_test_measures.pth.tar", # lambda 0.03
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-191308/rs_fairness/rs_apd_test_measures.pth.tar",  # lambda 0.05
#                             "./log/checkpoint_adult/checkpoint_adult-20210625-194213/rs_fairness/rs_apd_test_measures.pth.tar", # lambda 0.1
#
#                            # "./log/checkpoint_adult/checkpoint_adult-20210624-164148/rs_fairness/rs_apd_test_measures.pth.tar",
#                            # "./log/checkpoint_adult/checkpoint_adult-20210624-165505/rs_fairness/rs_apd_test_measures.pth.tar",
#                            # "./log/checkpoint_adult/checkpoint_adult-20210624-164810/rs_fairness/rs_apd_test_measures.pth.tar",
#                             ]

checkpoint_buf_buf = [checkpoint_fname_buf_vln, checkpoint_fname_buf_apd, checkpoint_fname_buf_rs]
marker_buf = ['^', 'o', 'v']

def main():


    for alg_index, checkpoint_fname_buf in enumerate(checkpoint_buf_buf):

        x_data = []
        y_data = []
        for chackpoint_fname in checkpoint_fname_buf:

            checkpoint_active = load_checkpoint(chackpoint_fname)
            test_measures_active = checkpoint_active["choose_test_measures_buffer"]

            # print(test_measures_active)

            mean_accuracy = np.array([[y['accuracy'] for y in x] for x in test_measures_active]).mean(axis=0)
            mean_equal_odds = np.array([[y['equal_odds'] for y in x] for x in test_measures_active]).mean(axis=0)

            x_data.append(mean_equal_odds[-1])
            y_data.append(mean_accuracy[-1])

        print(x_data)
        print(y_data)
        #

        marker = marker_buf[alg_index]
        plt.plot(x_data, y_data, marker=marker)

    # plt.figure()
    # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1000))
    plt.xlabel("Equality of Odds", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.grid()
    # plt.legend(loc='upper right', fontsize=20, frameon=False)
    # plt.savefig(os.path.join(hyperparam1["root_log_dir"], './FPR_GAP_PSS_vs_RS.png'))

    plt.show()


if __name__ == "__main__":
    main()
