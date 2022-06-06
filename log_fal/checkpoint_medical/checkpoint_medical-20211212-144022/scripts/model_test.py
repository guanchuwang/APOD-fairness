import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression


import cvxpy as cp
import numpy as np

from utils import load_problem_from_options, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_torch, estimate_objective
from utils import load_data, load_checkpoint, save_checkpoint


import random
from sklearn.metrics.pairwise import euclidean_distances
import json, os

from main_active_pss import mlp

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
        self.hidden_dim = hyperparam["hidden_dim"]
        self.layer_num = hyperparam["layer_num"]

    def load_data(self, run_num, checkpoint_dirname):

        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)

        print(self.x_train.shape, self.x_val.shape, self.x_test.shape)

        cur_path = os.getcwd()
        self.cp_path_clf = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf")
        self.cp_path_clfz = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clfz")
        self.label_budget = int(self.label_budget_percent * self.x_test.shape[0])

    def load_clfz(self):
        clfz_checkpoint = load_checkpoint(os.path.join(self.cp_path_clfz, os.listdir(self.cp_path_clfz)[0]))
        self.clf_z = clfz_checkpoint["clfz"]

    def load_clf(self, label_num):

        cp_name = "clf_" + str(label_num) + ".pth.tar"
        cp_fullname = os.path.join(self.cp_path_clf, cp_name)
        checkpoint = load_checkpoint(cp_fullname)

        self.weight = checkpoint["weight"]
        self.bias = checkpoint["bias"]

        self.clf_body = mlp(input_dim=checkpoint["clf_input_dim"], output_dim=checkpoint["clf_output_dim"], layer_num=self.layer_num, hidden_dim=self.hidden_dim)
        self.clf_body.load_state_dict(checkpoint["clf_state_dict"])

    def validate(self, hyperparam):
        z_prob_val = self.clf_z.predict_proba(self.x_val)
        estimate_measures = estimate_objective(self.x_val, self.y_val, z_prob_val, self.clf_body,
                                               self.weight, self.bias, self.problem, hyperparam)
        test_measures = measure_objective_results_torch(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                   self.weight, self.bias, self.problem, None)

        return estimate_measures, test_measures

    def validate_vallina(self):
        self.load_clf(label_num=0)
        test_measures = measure_objective_results_torch(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                        self.weight, self.bias, self.problem, None)
        return test_measures

# bank
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210515-212355/bank.json" # best pss
# test_hyperparam_fname = "./log/checkpoint_bank/checkpoint_bank-20210518-104747/bank.json" # 30 labels
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210515-220146/bank.json" # best rs
# test_hyperparam_fname = "./log_rs/checkpoint_bank/checkpoint_bank-20210518-105442/bank.json" # 30 labels

# adult
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210515-173437/adult.json"
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210515-181158/adult.json"
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210515-184121/adult.json"
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210515-200455/adult.json"
# test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210515-191828/adult.json" # Best

test_hyperparam_fname = "./log/checkpoint_adult/checkpoint_adult-20210623-144723/adult.json" # Best

# test_hyperparam_fname = "./log_rs/checkpoint_adult/checkpoint_adult-20210515-201732/adult.json"

# ucla_law_race
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-165448/ucla_law_race.json" # K=40
# test_hyperparam_fname = "./log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-163110/ucla_law_race.json" # K=20 # best
# test_hyperparam_fname = "./log_rs/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-171540/ucla_law_race.json" # K=40
# test_hyperparam_fname = "./log_rs/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210518-164201/ucla_law_race.json" # K=20

# test_hyperparam_fname = "./log/checkpoint_region_job_n/checkpoint_region_job_n-20210512-161132/region_job_n.json"
# test_hyperparam_fname = "./log_rs/checkpoint_region_job_n/checkpoint_region_job_n-20210512-165627/region_job_n.json"

### Medical
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210515-144849/medical.json"
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210515-151442/medical.json"
# test_hyperparam_fname = "./log/checkpoint_medical/checkpoint_medical-20210515-153212/medical.json" # best pss
#
# test_hyperparam_fname = "./log_rs/checkpoint_medical/checkpoint_medical-20210515-154807/medical.json" # best rs




# region_job
# test_hyperparam_fname = "./log/checkpoint_region_job_z/checkpoint_region_job_z-20210512-174711/region_job_z.json"
# test_hyperparam_fname = "./log/checkpoint_region_job_z/checkpoint_region_job_z-20210512-185156/region_job_z.json"
# test_hyperparam_fname = "./log_rs/checkpoint_region_job_z/checkpoint_region_job_z-20210512-194411/region_job_z.json"


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
for run_num in range(2): # total_run_num):

    agent = Test_agent(problem, hyperparam)
    agent.load_data(run_num, hyperparam["root_log_dir"])

    # validate vallina model
    vallina_measures = agent.validate_vallina()
    vallina_measures_buffer.append(vallina_measures)

    # validate debiased model
    agent.load_clfz()
    estimate_measures_buf = []
    test_measures_buf = []
    for label_num in range(hyperparam["warmup_labels"]*2+1, agent.label_budget):
        agent.load_clf(label_num)
        estimate_measures, test_measures = agent.validate(hyperparam)
        estimate_measures_buf.append(estimate_measures)
        test_measures_buf.append(test_measures)

    best_estimate_measures_buf = []
    choose_test_measures_buf = []
    for label_num in range(agent.label_budget - hyperparam["warmup_labels"]*2):
        object_buf = torch.tensor([estimate_result['objective'] for estimate_result in estimate_measures_buf])
        best_index = object_buf[:label_num+1].argmax()
        # best_index = label_num
        best_estimate_measures_buf.append(estimate_measures_buf[best_index])
        choose_test_measures_buf.append(test_measures_buf[best_index])

    estimate_measures_buffer.append(estimate_measures_buf)
    test_measures_buffer.append(test_measures_buf)
    best_estimate_measures_buffer.append(best_estimate_measures_buf)
    choose_test_measures_buffer.append(choose_test_measures_buf)

save_checkpoint(hyperparam["root_log_dir"] + "/test_measures.pth.tar",
                estimate_measures_buffer=estimate_measures_buffer,
                test_measures_buffer=test_measures_buffer,
                best_estimate_measures_buffer=best_estimate_measures_buffer,
                choose_test_measures_buffer=choose_test_measures_buffer)


mean_vallina_accuracy = np.array([x['accuracy'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_objective = np.array([x['objective'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_fpr_diff = np.array([x['fpr_diff'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_fnr_diff = np.array([x['fnr_diff'] for x in vallina_measures_buffer]).mean(axis=0)

mean_vallina_fnr1 = np.array([x['1_fnr'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_fnr0 = np.array([x['0_fnr'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_fpr1 = np.array([x['1_fpr'] for x in vallina_measures_buffer]).mean(axis=0)
mean_vallina_fpr0 = np.array([x['0_fpr'] for x in vallina_measures_buffer]).mean(axis=0)

print(mean_vallina_fnr1 - mean_vallina_fnr0,
        mean_vallina_fpr1 - mean_vallina_fpr0,
        mean_vallina_fnr1 - mean_vallina_fnr0 + (mean_vallina_fpr1 - mean_vallina_fpr0))

mean_accuracy = np.array([[y['accuracy'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
mean_objective = np.array([[y['objective'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
mean_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)
mean_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in choose_test_measures_buffer]).mean(axis=0)

print("Mean Vallina result:")
print("Acc", mean_vallina_accuracy)
print("Fpr", mean_vallina_fpr_diff)
print("Fnr", mean_vallina_fnr_diff)
print("Obj", mean_vallina_objective)
print("----------------------------")

print("Mean Best Debiased result:")
print("Acc", mean_accuracy)
print("Fpr", mean_fpr_diff)
print("Fnr", mean_fnr_diff)
print("Obj", mean_objective)
print("----------------------------")

# print(np.array([[y['fpr_diff'] for y in x] for x in choose_test_measures_buffer]))
print(np.array([[y['fnr_diff'] for y in x] for x in choose_test_measures_buffer]))
