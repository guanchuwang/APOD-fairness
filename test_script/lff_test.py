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
from utils import measure_objective_results_body_head
from utils import load_data, load_checkpoint, save_checkpoint
# 

import random
from sklearn.metrics.pairwise import euclidean_distances
import json, os

from model import mlp

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Test_agent:

    def __init__(self, problem, hyperparam):
        self.problem = problem
        self.embedding_dim = hyperparam["embedding_dim"]
        self.label_budget_percent = hyperparam["label_budget_percent"]
        # self.hidden_dim = hyperparam["hidden_dim"]
        # self.layer_num = hyperparam["layer_num"]

    def load_data(self, run_num, checkpoint_dirname):

        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)

        print(self.x_train.shape, self.x_val.shape, self.x_test.shape)

        cur_path = os.getcwd()
        self.cp_path_clfb = os.path.join(cur_path, "../", checkpoint_dirname, "round" + str(run_num), "clf_body")
        self.cp_path_clfh = os.path.join(cur_path, "../", checkpoint_dirname, "round" + str(run_num), "clf_head")

    def load_clf_body(self):

        # print(os.listdir(self.cp_path_clfb))
        clfb_checkpoint = load_checkpoint(os.path.join(self.cp_path_clfb, os.listdir(self.cp_path_clfb)[0]))
        self.clf_body = mlp(input_dim=clfb_checkpoint["clf_input_dim"], output_dim=clfb_checkpoint["clf_output_dim"],
                            layer_num=clfb_checkpoint["layer_num"], hidden_dim=clfb_checkpoint["hidden_dim"])
        self.clf_body.load_state_dict(clfb_checkpoint["clf_body_state_dict"])
        # print(self.clf_body)

    def load_clf_head(self):

        cp_name = "clf_head.pth.tar"
        cp_fullname = os.path.join(self.cp_path_clfh, cp_name)
        checkpoint = load_checkpoint(cp_fullname)

        self.clf_head = mlp(input_dim=checkpoint["clf_input_dim"], output_dim=checkpoint["clf_output_dim"],
                            layer_num=checkpoint["layer_num"], hidden_dim=checkpoint["hidden_dim"])

        self.clf_head.load_state_dict(checkpoint["clf_head_state_dict"])

    def validate(self):
        test_measures = measure_objective_results_body_head(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                            self.clf_head, self.problem, None)

        # print(test_measures['accuracy'], test_measures['fpr_diff'], test_measures['fnr_diff'])

        return test_measures

    # def validate_vallina(self):
    #     self.load_clf_body()
    #     self.load_clf_head(label_num=0)
    #     test_measures = measure_objective_results_body_head_eop(self.x_test, self.y_test, self.z_test, self.clf_body,
    #                                                         self.clf_head, self.problem, None)
    #     return test_measures



### Medical LFF
test_hyperparam_fname = "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145354/medical_lff.json" # q=2.5, obj_acc=1, epoch=50 body_hidden_dim 128 head_hidden_dim 128
# test_hyperparam_fname = "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145515/medical_lff.json" # q=2.6, obj_acc=1, epoch=50 body_hidden_dim 128 head_hidden_dim 128
# test_hyperparam_fname = "../log_lff/checkpoint_medical/checkpoint_medical-20211213-145849/medical_lff.json" # q=2.9, obj_acc=1, epoch=50 body_hidden_dim 128 head_hidden_dim 128
# test_hyperparam_fname = "../log_lff/checkpoint_medical/checkpoint_medical-20211213-150351/medical_lff.json" # q=2.94, obj_acc=1, epoch=50 body_hidden_dim 128 head_hidden_dim 128


# # Medical DRO
# test_hyperparam_fname = "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-173918/medical_DRO.json" # obj_acc 5 debias_alpha 0.9, debias_gamma 0.1 epoch=50   body_hidden_dim 128 head_hidden_dim =128 embedding_dim 64
# test_hyperparam_fname = "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-174151/medical_DRO.json" # obj_acc 10 debias_alpha 0.91, debias_gamma 0.1 epoch=50   body_hidden_dim 128 head_hidden_dim =128 embedding_dim 64
# test_hyperparam_fname = "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-174930/medical_DRO.json" # obj_acc 10 debias_alpha 0.97, debias_gamma 0.1 epoch=50   body_hidden_dim 128 head_hidden_dim =128 embedding_dim 64
# test_hyperparam_fname = "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-181629/medical_DRO.json" # obj_acc 100 debias_alpha 0.999999, debias_gamma 0.1 epoch=50   body_hidden_dim 128 head_hidden_dim =128 embedding_dim 128
# test_hyperparam_fname = "../log_DRO/checkpoint_medical/checkpoint_medical-20211215-182754/medical_DRO.json" # obj_acc 100 debias_alpha 0.999999, debias_gamma 0.1 epoch=50   body_hidden_dim 128 head_hidden_dim =128 embedding_dim 128



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
        agent.load_clf_body()
        agent.load_clf_head()
        test_measures = agent.validate()
        test_measures_buffer.append([test_measures])

    save_checkpoint("../" + hyperparam["root_log_dir"] + "/test_measures.pth.tar",
                    choose_test_measures_buffer=test_measures_buffer,
                    )



    mean_accuracy = np.array([x[0]['accuracy'] for x in test_measures_buffer]).mean(axis=0)
    mean_objective = np.array([x[0]['objective'] for x in test_measures_buffer]).mean(axis=0)
    mean_fpr_diff = np.array([x[0]['fpr_diff'] for x in test_measures_buffer]).mean(axis=0)
    mean_fnr_diff = np.array([x[0]['fnr_diff'] for x in test_measures_buffer]).mean(axis=0)
    mean_fnr0 = np.array([x[0]['0_fnr'] for x in test_measures_buffer]).mean(axis=0)
    mean_fnr1 = np.array([x[0]['1_fnr'] for x in test_measures_buffer]).mean(axis=0)
    mean_eo = -mean_fpr_diff - mean_fnr_diff
    print(mean_fnr0, mean_fnr1, mean_fnr0, mean_fnr1)
    mean_eop = min(1-mean_fnr0, 1-mean_fnr1) / max(1-mean_fnr0, 1-mean_fnr1)

    print("Mean Best Debiased result:")
    print("Acc", mean_accuracy)
    print("Fpr", mean_fpr_diff)
    print("Fnr", mean_fnr_diff)
    print("EO", mean_eo)
    print("EOP", mean_eop)

    # print("Obj", mean_objective)
    print("----------------------------")

    # print(np.array([[y['accuracy'] for y in x] for x in choose_test_measures_buffer]))
    # print(np.array([[y['fpr_diff'] for y in x] for x in choose_test_measures_buffer]))
    # print(np.array([[y['fnr_diff'] for y in x] for x in choose_test_measures_buffer]))
