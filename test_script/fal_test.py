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

import sys
sys.path.append("../")

from utils import load_problem_from_options, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_body_head, estimate_objective_body_head
from utils import load_data, load_checkpoint, save_checkpoint
# 

import random
from sklearn.metrics.pairwise import euclidean_distances
import json, os

from model import mlp
from apd_test_eop import Test_agent as APDTest_agent

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

        # print(checkpoint_dirname)
        cur_path = os.getcwd()
        # self.cp_path_clfb = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf_body")
        self.cp_path_clfh = os.path.join(cur_path, "../", checkpoint_dirname, "round" + str(run_num), "clf_head")
        self.cp_path_clfz = os.path.join(cur_path, "../", checkpoint_dirname, "round" + str(run_num), "clfz")
        self.label_budget = int(self.label_budget_percent * self.x_test.shape[0])

    def load_clf_head(self, label_num):

        cp_name = "clf_head_" + str(label_num) + ".pth.tar"
        cp_fullname = os.path.join(self.cp_path_clfh, cp_name)
        # print(cp_fullname)
        checkpoint = load_checkpoint(cp_fullname)
        self.clf_head = checkpoint["clf_head"].predict_proba

    def validate(self):
        self.load_clf_head(label_num=self.label_budget)
        test_measures = measure_objective_results_body_head(self.x_test, self.y_test, self.z_test, None,
                                                            self.clf_head, self.problem, None)
        return test_measures

test_hyperparam_fname = ""


### Medical FAL
test_hyperparam_fname = "../log_fal/checkpoint_medical/checkpoint_medical-20211212-144022/medical_fal.json" # alpha 0.6  seed 0 active_buf_size 100
# test_hyperparam_fname = "../log_fal/checkpoint_medical/checkpoint_medical-20211212-212056/medical_fal.json" # alpha 0.9 active_buf_size 2
# test_hyperparam_fname = "../log_fal/checkpoint_medical/checkpoint_medical-20211212-212353/medical_fal.json" # alpha 0.8 active_buf_size 2


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

        # vallina_measures = agent.validate_vallina()
        # vallina_measures_buffer.append([vallina_measures])

        baseline_measures = agent.validate()
        baseline_measures_buffer.append([baseline_measures])

    print(hyperparam["root_log_dir"])
    # save_checkpoint(hyperparam["root_log_dir"] + "/vallina_measures.pth.tar",
    #                 choose_test_measures_buffer=vallina_measures_buffer)
    save_checkpoint("../" + hyperparam["root_log_dir"] + "/test_measures.pth.tar",
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
