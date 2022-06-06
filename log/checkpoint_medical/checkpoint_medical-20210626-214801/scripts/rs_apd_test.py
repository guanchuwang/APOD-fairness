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

    def validate_rs_apd(self):
        self.load_clf_body()
        self.load_clf_head(label_num=self.label_budget)
        test_measures = measure_objective_results_body_head(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                            self.clf_head, self.problem, None)
        return test_measures

# bank
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210626-211903/rs_fairness/bank.json" # 30 lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210626-211400/rs_fairness/bank.json" # 30 lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151542/rs_fairness/bank.json" # 30 lambda 0.1
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151659/rs_fairness/bank.json" # 30 lambda 0.5
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210624-151336/rs_fairness/bank.json" # 30 lambda 1.

# ucla_law_race
# test_hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-210855/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.0005
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-161439/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162952/rs_fairness/ucla_law_race_preprocess.json" # lambda 0.005


### Medical
test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-114757/rs_fairness/medical.json" # apd lambda 0.001
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-115807/rs_fairness/medical.json" # apd lambda 0.005
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-114335/rs_fairness/medical.json" # best apd lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210624-115243/rs_fairness/medical.json" # apd lambda 0.1


# adult
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-164148/rs_fairness/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-165505/rs_fairness/adult_preprocess.json" # lambda 0.05
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210624-164810/rs_fairness/adult_preprocess.json" # lambda 0.1

# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-192429/rs_fairness/adult_preprocess.json" # lambda 0.01
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-211903/rs_fairness/adult_preprocess.json" # lambda 0.02
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-212415/rs_fairness/adult_preprocess.json" # lambda 0.03
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-191308/rs_fairness/adult_preprocess.json" # lambda 0.05
# test_hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-194213/rs_fairness/adult_preprocess.json" # lambda 0.1

if __name__ == "__main__":

    with open(test_hyperparam_fname) as json_file:
        hyperparam = json.load(json_file)

    setup_seed(hyperparam["seed"])

    problem = load_problem_from_options(hyperparam['problem_options'])

    rs_apd_measures_buffer = []
    vallina_measures_buffer = []

    total_run_num = hyperparam["round_num"]
    for run_num in range(total_run_num):

        agent = Test_agent(problem, hyperparam)
        agent.load_data(run_num, hyperparam["root_log_dir"])

        vallina_measures = agent.validate_vallina()
        vallina_measures_buffer.append([vallina_measures])

        rs_apd_measures = agent.validate_rs_apd()
        rs_apd_measures_buffer.append([rs_apd_measures])

    save_checkpoint(hyperparam["root_log_dir"] + "/rs_apd_test_measures.pth.tar",
                    choose_test_measures_buffer=rs_apd_measures_buffer)

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

    mean_rs_apd_accuracy = np.array([[y['accuracy'] for y in x] for x in rs_apd_measures_buffer]).mean(axis=0)
    mean_rs_apd_objective = np.array([[y['objective'] for y in x] for x in rs_apd_measures_buffer]).mean(axis=0)
    mean_rs_apd_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in rs_apd_measures_buffer]).mean(axis=0)
    mean_rs_apd_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in rs_apd_measures_buffer]).mean(axis=0)
    mean_rs_apd_equal_odds = np.array([[y['equal_odds'] for y in x] for x in rs_apd_measures_buffer]).mean(axis=0)

    print("Mean RS APD result:")
    print("Acc", mean_rs_apd_accuracy)
    print("Fpr", mean_rs_apd_fpr_diff)
    print("Fnr", mean_rs_apd_fnr_diff)
    print("EO",  mean_rs_apd_equal_odds)
    print("Obj", mean_rs_apd_objective)
    print("----------------------------")
