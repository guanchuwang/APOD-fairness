import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression


# import cvxpy as cp
import numpy as np

from utils import load_problem_from_options, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_body_head, estimate_objective_body_head
from utils import load_data, load_checkpoint, save_checkpoint
#

import random
from sklearn.metrics.pairwise import euclidean_distances
import json, os

from main_active_pss import mlp
from apd_test import Test_agent as APDTest_agent

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Test_agent(APDTest_agent):

    def __init__(self, problem, hyperparam):
        super().__init__(problem, hyperparam)

    def load_data(self, run_num, checkpoint_dirname):

        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)

        print(self.x_train.shape, self.x_val.shape, self.x_test.shape)

        cur_path = os.getcwd()
        self.cp_path_clfb = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf_body")
        self.cp_path_clfh = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf_head")
        self.cp_path_clfz = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clfz")
        self.label_budget = int(self.label_budget_percent * self.x_test.shape[0])

    def validate(self):
        self.load_clf_body()
        self.load_clf_head(label_num=self.label_budget)
        test_measures = measure_objective_results_body_head(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                            self.clf_head, self.problem, None)
        return test_measures


# bank RS (Both are necessary)

# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210626-211903/rs_fairness/bank.json" # 30 lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210626-211400/rs_fairness/bank.json" # 30 lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151542/rs_fairness/bank.json" # 30 lambda 0.1
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151659/rs_fairness/bank.json" # 30 lambda 0.5
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151336/rs_fairness/bank.json" # 30 lambda 1.

# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-163056/bank.json" # lambda 1.
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-163408/bank.json" # lambda 0.1
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-163645/bank.json" # lambda 0.2
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-163834/bank.json" # lambda 0.3
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-164002/bank.json" # lambda 1.2
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-164214/bank.json" # lambda 2
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-164747/bank.json" # lambda 3
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-164423/bank.json"  # lambda 5
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210821-164321/bank.json" # lambda 10




### Medical RS
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-114757/rs_fairness/medical.json" # apd lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-115807/rs_fairness/medical.json" # apd lambda 0.005
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-114335/rs_fairness/medical.json" # best apd lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210626-215613/rs_fairness/medical.json" # best apd lambda 0.02
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210626-215334/rs_fairness/medical.json" # best apd lambda 0.05
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-115243/rs_fairness/medical.json" # apd lambda 0.1
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210626-214032/rs_fairness/medical.json" # apd lambda 0.2


# adult RS
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-164148/rs_fairness/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-165505/rs_fairness/adult_preprocess.json" # lambda 0.05
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-164810/rs_fairness/adult_preprocess.json" # lambda 0.1

# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-192429/rs_fairness/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-211903/rs_fairness/adult_preprocess.json" # lambda 0.02
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-212415/rs_fairness/adult_preprocess.json" # lambda 0.03
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-191308/rs_fairness/adult_preprocess.json" # lambda 0.05
test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-194213/rs_fairness/adult_preprocess.json" # lambda 0.1


# default RS
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-171653/default.json" # lambda 0.02
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-172025/default.json" # lambda 0.06
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-171803/default.json" # lambda 0.1
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-172404/default.json" # lambda 0.2
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-172233/default.json" # lambda 0.5
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210706-162659/default.json" # lambda 1


# default US
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-172545/default.json" # lambda 0.2
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-173111/default.json" # lambda 1
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-181606/default.json" # lambda 2
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-173550/default.json" # lambda 5
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-175040/default.json" # lambda 6
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-182154/default.json" # lambda 7
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-174532/default.json" # lambda 8
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-174021/default.json" # lambda 10

# default CSA
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-194638/default.json" # lambda 0.0001
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-194044/default.json" # lambda 0.001
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-193521/default.json" # lambda 0.01
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-195715/default.json" # lambda 0.02
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-195211/default.json" # lambda 0.05
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-193025/default.json" # lambda 0.1
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-192525/default.json" # lambda 1


## bank US
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-175653/bank.json" # lambda 0.2
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-175531/bank.json" # lambda 0.4
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-175929/bank.json" # lambda 0.5
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-173643/bank.json" # lambda 0.6
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-175303/bank.json" # lambda 0.7
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-174905/bank.json" # lambda 0.8
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-175111/bank.json" # lambda 0.9
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-174018/bank.json" # lambda 1
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-174505/bank.json" # lambda 1.1
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-174311/bank.json" # lambda 1.5
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-174121/bank.json" # lambda 2
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210630-180046/bank.json" # lambda 5
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210821-170336/bank.json" # lambda 8
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210821-170747/bank.json" # lambda 9
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210821-170919/bank.json" # lambda 10
# test_hyperparam_fname = "log_us/checkpoint_bank/checkpoint_bank-20210821-170614/bank.json" # lambda 12


## Medical US
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-113135/medical.json" # lambda 0.001
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-112837/medical.json" # lambda 0.005
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-112512/medical.json" # lambda 0.01
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-114003/medical.json" # lambda 0.015
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-112034/medical.json" # lambda 0.02
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-114307/medical.json" # lambda 0.025
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210630-113643/medical.json" # lambda 0.05


## adult US
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-095502/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-094406/adult_preprocess.json" # lambda 0.06
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-100522/adult_preprocess.json" # lambda 0.07
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-101144/adult_preprocess.json" # lambda 0.08
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-102412/adult_preprocess.json" # lambda 0.09
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210630-095954/adult_preprocess.json" # lambda 0.1


## bank CSA
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-180522/bank.json" # lambda 0.01
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-180424/bank.json" # lambda 0.1
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-181054/bank.json" # lambda 0.15
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-180732/bank.json" # lambda 0.2
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-180909/bank.json" # lambda 0.3
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-181544/bank.json" # lambda 0.35
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-181312/bank.json" # lambda 0.4
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-181213/bank.json" # lambda 0.45
# test_hyperparam_fname = "log_csa/checkpoint_bank/checkpoint_bank-20210630-180250/bank.json" # lambda 0.5


## Medical CSA
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-115306/medical.json" # lambda 0.0001
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-115745/medical.json" # lambda 0.0005
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-120106/medical.json" # lambda 0.001
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-120405/medical.json" # lambda 0.005
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-120723/medical.json" # lambda 0.01
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-121048/medical.json" # lambda 0.02
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-121434/medical.json" # lambda 0.03
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210630-121757/medical.json" # lambda 0.05


## adult CSA
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-171203/adult_preprocess.json" # lambda 0.00001
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-171648/adult_preprocess.json" # lambda 0.00005
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-170503/adult_preprocess.json" # lambda 0.0001
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-163027/adult_preprocess.json" # lambda 0.001
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-165111/adult_preprocess.json" # lambda 0.005
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-163553/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-173103/adult_preprocess.json" # lambda 0.05
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-172211/adult_preprocess.json" # lambda 0.1



if __name__ == "__main__":

    with open(test_hyperparam_fname) as json_file:
        hyperparam = json.load(json_file)

    setup_seed(hyperparam["seed"])

    problem = load_problem_from_options(hyperparam['problem_options'])

    baseline_measures_buffer = []
    vallina_measures_buffer = []

    total_run_num = hyperparam["round_num"]
    for run_num in range(total_run_num):

        agent = Test_agent(problem, hyperparam)
        agent.load_data(run_num, hyperparam["root_log_dir"])

        vallina_measures = agent.validate_vallina()
        vallina_measures_buffer.append([vallina_measures])

        baseline_measures = agent.validate()
        baseline_measures_buffer.append([baseline_measures])

    print(hyperparam["root_log_dir"])
    save_checkpoint(hyperparam["root_log_dir"] + "/vallina_measures.pth.tar",
                    choose_test_measures_buffer=vallina_measures_buffer)
    save_checkpoint(hyperparam["root_log_dir"] + "/test_measures.pth.tar",
                    choose_test_measures_buffer=baseline_measures_buffer)

    mean_vallina_accuracy = np.array([[y['accuracy'] for y in x] for x in vallina_measures_buffer]).mean(axis=0)
    mean_vallina_objective = np.array([[y['objective'] for y in x] for x in vallina_measures_buffer]).mean(axis=0)
    mean_vallina_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in vallina_measures_buffer]).mean(axis=0)
    mean_vallina_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in vallina_measures_buffer]).mean(axis=0)
    mean_vallina_equal_odds = np.array([[y['equal_odds'] for y in x] for x in vallina_measures_buffer]).mean(axis=0)

    print("Mean Vallina result:")
    print("Acc", mean_vallina_accuracy)
    print("Fpr", mean_vallina_fpr_diff)
    print("Fnr", mean_vallina_fnr_diff)
    print("EO", mean_vallina_equal_odds)
    print("Obj", mean_vallina_objective)
    print("----------------------------")

    mean_baseline_accuracy = np.array([[y['accuracy'] for y in x] for x in baseline_measures_buffer]).mean(axis=0)
    mean_baseline_objective = np.array([[y['objective'] for y in x] for x in baseline_measures_buffer]).mean(axis=0)
    mean_baseline_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in baseline_measures_buffer]).mean(axis=0)
    mean_baseline_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in baseline_measures_buffer]).mean(axis=0)
    mean_baseline_equal_odds = np.array([[y['equal_odds'] for y in x] for x in baseline_measures_buffer]).mean(axis=0)

    print("Mean RS APD result:")
    print("Acc", mean_baseline_accuracy)
    print("Fpr", mean_baseline_fpr_diff)
    print("Fnr", mean_baseline_fnr_diff)
    print("EO",  mean_baseline_equal_odds)
    print("Obj", mean_baseline_objective)
    print("----------------------------")



# ucla_law_race RS
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-210855/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0001
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-111653/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0002
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-111558/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0003
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-110815/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0004
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0005
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-161439/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162952/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.005

# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-160932/rs_fairness/ucla_law_race_preprocess.json" # hidden_dim 16 lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-102911/rs_fairness/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 32 lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-104218/rs_fairness/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 64 lambda 0.0001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-104829/rs_fairness/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 64 lambda 0.0005
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-105945/rs_fairness/ucla_law_race_preprocess.json" # embedding_dim 16 hidden_dim 16 lambda 0.0001

# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-112036/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 32 hidden_dim 32 lambda 0.0001
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-205407/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 32 hidden_dim 32 lambda 0.001
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-210734/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 32 hidden_dim 32 lambda 0.0008
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-211324/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 32 hidden_dim 32 lambda 0.0006
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-211823/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 32 hidden_dim 32 lambda 0.0004
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-212540/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0005
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-213040/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0007
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-213746/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0002
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-214300/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.001
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-220114/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.002
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210628-105905/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0015
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210628-110220/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0017
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210628-110402/rs_fairness/ucla_law_race_preprocess.json" # seed 0 embedding_dim 64 hidden_dim 32 lambda 0.0018

## ucla_law_race US
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-195948/ucla_law_race_preprocess.json" # lambda 0.0001
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-200733/ucla_law_race_preprocess.json" # lambda 0.0005
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-200344/ucla_law_race_preprocess.json" # lambda 0.001
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-195540/ucla_law_race_preprocess.json" # lambda 0.0018
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-202230/ucla_law_race_preprocess.json" # lambda 0.003
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-201725/ucla_law_race_preprocess.json" # lambda 0.005
# test_hyperparam_fname = "log_us/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-201402/ucla_law_race_preprocess.json" # lambda 0.01

## ucla_law_race
# test_hyperparam_fname = "log_csa/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210630-205200/ucla_law_race_preprocess.json" # lambda 0.0015
