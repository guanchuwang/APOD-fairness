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

    # def load_data(self, run_num, checkpoint_dirname):
    #
    #     self.x_train, self.x_val, self.x_test, \
    #     self.y_train, self.y_val, self.y_test, \
    #     self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)
    #
    #     print(self.x_train.shape, self.x_val.shape, self.x_test.shape)
    #
    #     cur_path = os.getcwd()
    #     self.cp_path_clfb = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf_body")
    #     self.cp_path_clfh = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf_head")
    #     self.cp_path_clfz = os.path.join("./log/checkpoint_medical/checkpoint_medical-20210626-214801", "round" + str(run_num), "clfz")
    #     self.label_budget = int(self.label_budget_percent * self.x_test.shape[0])

    def validate(self, hyperparam):

        test_measures = measure_objective_results_body_head(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                            self.clf_head, self.problem, None)

        return test_measures


### Medical RS
# test_hyperparam_fname = "./log_rs/checkpoint_medical/checkpoint_medical-20210714-115219/medical.json" # apd lambda 0.2

## Medical US
# test_hyperparam_fname = "log_us/checkpoint_medical/checkpoint_medical-20210714-105607/medical.json" # lambda 0.15


## Medical CSA
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210714-100133/medical.json" # lambda 0.15
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210714-112057/medical.json" # lambda 0.2
# test_hyperparam_fname = "log_csa/checkpoint_medical/checkpoint_medical-20210714-112501/medical.json" # lambda 0.5

## Medical Ablation Individual
# test_hyperparam_fname = "ablation/log/checkpoint_medical/checkpoint_medical-20210721-174959/medical_ablation.json" # lambda 0.15

## Adult RS
# test_hyperparam_fname = "log_rs/checkpoint_adult/checkpoint_adult-20210714-143928/adult_preprocess.json" # lambda 0.5

## Adult US
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210714-143314/adult_preprocess.json" # lambda 0.03
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210714-125136/adult_preprocess.json" # lambda 0.5
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210714-140821/adult_preprocess.json" # lambda 5
# test_hyperparam_fname = "log_us/checkpoint_adult/checkpoint_adult-20210714-140234/adult_preprocess.json" # lambda 5

## Adult CSA
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210630-173103/adult_preprocess.json" # lambda 0.05
# test_hyperparam_fname = "log_csa/checkpoint_adult/checkpoint_adult-20210714-144847/adult_preprocess.json" # lambda 0.5


# default RS
# test_hyperparam_fname = "log_rs/checkpoint_default/checkpoint_default-20210714-164224/default.json" # lambda 0.02


# default US
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210714-162755/default.json" # lambda 0.02
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210714-163621/default.json" # lambda 0.1
# test_hyperparam_fname = "log_us/checkpoint_default/checkpoint_default-20210706-172545/default.json" # lambda 0.2


# default CSA
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-195715/default.json" # lambda 0.02
# test_hyperparam_fname = "log_csa/checkpoint_default/checkpoint_default-20210706-193025/default.json" # lambda 0.1

### Default Ablation Individual
test_hyperparam_fname = "./ablation/log/checkpoint_default/checkpoint_default-20210721-180059/default_ablation.json" # lambda 0.1


if __name__ == "__main__":

    with open(test_hyperparam_fname) as json_file:
        hyperparam = json.load(json_file)

    setup_seed(hyperparam["seed"])

    problem = load_problem_from_options(hyperparam['problem_options'])

    vallina_measures_buffer = []
    estimate_measures_buffer = []
    test_measures_buffer = []
    best_estimate_measures_buffer = []
    choose_test_measures_buffer = []

    total_run_num = hyperparam["round_num"]
    for run_num in range(total_run_num):

        agent = Test_agent(problem, hyperparam)

        # print(hyperparam["root_log_dir"])
        agent.load_data(run_num, hyperparam["root_log_dir"])

        # validate vallina model
        vallina_measures = agent.validate_vallina()
        vallina_measures_buffer.append([vallina_measures])

        # validate debiased model
        choose_test_measures_buf = []
        for label_num in range(hyperparam["warmup_labels"]*2+1, agent.label_budget+1):
            agent.load_clf_head(label_num)
            test_measures = agent.validate(hyperparam)
            choose_test_measures_buf.append(test_measures)

        choose_test_measures_buffer.append(choose_test_measures_buf)

    save_checkpoint(hyperparam["root_log_dir"] + "/test_measures_ts.pth.tar",
                    x_data=list(np.arange(hyperparam["warmup_labels"]*2+1, agent.label_budget+1)/agent.x_test.shape[0]),
                    choose_test_measures_buffer=choose_test_measures_buffer)

    # save_checkpoint(hyperparam["root_log_dir"] + "/vallina_measures.pth.tar",
    #                 choose_test_measures_buffer=vallina_measures_buffer)

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

    mean_accuracy = np.array([[y['accuracy'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
    mean_objective = np.array([[y['objective'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
    mean_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
    mean_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
    mean_equal_odds = np.array([[y['equal_odds'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)

    print("Mean Best Debiased result:")
    print("Acc", mean_accuracy)
    print("Fpr", mean_fpr_diff)
    print("Fnr", mean_fnr_diff)
    print("EO", mean_equal_odds)
    print("Obj", mean_objective)
    print("----------------------------")

