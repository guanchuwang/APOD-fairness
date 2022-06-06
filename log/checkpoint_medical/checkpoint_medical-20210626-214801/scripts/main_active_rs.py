import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


import cvxpy as cp
import numpy as np

from utils import load_problem_from_options, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_torch
from utils import load_data, save_checkpoint


import random
from sklearn.metrics.pairwise import euclidean_distances
import os, shutil

from utils import setup_seed, log_init
from component import PUA_agent

class RS_Agent(PUA_agent):

    def __init__(self, problem, hyperparam, logger):
        super().__init__(problem, hyperparam, logger)

    def generate_sensitive_label(self):
        unlabel_index = [index for index in range(self.x_train.shape[0]) if not self.if_label[index]]
        index = np.random.choice(unlabel_index, 1)
        self.if_label[index] = True

        self.clfz_fit()

        index00 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 0 and self.y_train[index] == 0]
        index01 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 0 and self.y_train[index] == 1]
        index10 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 1 and self.y_train[index] == 0]
        index11 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 1 and self.y_train[index] == 1]

        self.logger.info("Current 00 number %s, 01 number %s, 10 number %s, 11 number %s,", len(index00), len(index01), len(index10), len(index11))

        if (len(index00) < 1 or len(index10) < 1) and (len(index01) < 1 or len(index11) < 1):
            return False
        else:
            return True

    def clfz_fit(self):
        self.clf_z.fit(self.x_train[self.if_label], self.z_train[self.if_label])
        z_prob_train = self.clf_z.predict_proba(self.x_train)
        auc_z = metrics.roc_auc_score(self.z_train, z_prob_train[:, 1])
        self.logger.info('Z AUC = %s,', auc_z)


import argparse
parser = argparse.ArgumentParser("PSS")
parser.add_argument('--dataset', type=str, default='adult', help='.')
args = parser.parse_args()

def main():

    json_fname = args.dataset + ".json"

    import json
    # with open("./hyperparameters/adult.json") as json_file: # completed
    # with open("./hyperparameters/compas.json") as json_file:
    # with open("./hyperparameters/ucla_law_race.json") as json_file: # completed
    # with open("./hyperparameters/ucla_law_gender.json") as json_file:
    # with open("./hyperparameters/loan_default.json") as json_file:
    # with open("./hyperparameters/region_job.json") as json_file: # completed
    # with open("./hyperparameters/nba.json") as json_file:
    with open(os.path.join("hyperparameters", json_fname)) as json_file:
        hyperparam = json.load(json_file)
    hyperparam["log_rootdir"] = "log_rs"

    setup_seed(hyperparam["seed"])
    logger = log_init(hyperparam)

    # save hyperparameters
    with open(os.path.join(hyperparam["root_log_dir"], json_fname), "w") as json_file:
        json.dump(hyperparam, json_file)

    problem = load_problem_from_options(hyperparam['problem_options'])

    for run_num in range(hyperparam["round_num"]):
        agent = RS_Agent(problem, hyperparam, logger)
        agent.load_data(run_num=run_num)
        agent.fit(label_budget=hyperparam["label_num"])


if __name__ == "__main__":

    main()
