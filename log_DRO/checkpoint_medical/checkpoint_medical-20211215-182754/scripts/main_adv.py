import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# import cvxpy as cp
import numpy as np

from utils import load_problem_from_options, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_body_head
from utils import load_data, save_checkpoint
#

import random
from sklearn.metrics.pairwise import euclidean_distances
import os, shutil

# class mlp(nn.Module):
#
#     def __init__(self, input_dim, output_dim, hidden_dim=64):
#         super().__init__()
#         self.layer1 = nn.Linear(input_dim, output_dim)
#         self.layer1.weight.data.mul_(1e-3)
#         nn.init.constant_(self.layer1.bias.data, 0.)
#
#     def forward(self, x):
#         return self.layer1(x)


class mlp(nn.Module):

    def __init__(self, input_dim, output_dim, layer_num=1, hidden_dim=64, activation=None):
        super().__init__()

        self.mlp = nn.ModuleList()
        self.layer_num = layer_num
        self.activation = activation
        self.dropout = nn.Dropout(p=0.5)

        if layer_num == 1:
            layer1 = nn.Linear(input_dim, output_dim)
            layer1.weight.data.mul_(1e-3)
            nn.init.constant_(layer1.bias.data, 0.)
            self.mlp.append(layer1)

        else:
            for layer_index in range(layer_num):
                if layer_index == 0:
                    layer1 = nn.Linear(input_dim, hidden_dim)
                elif layer_index == layer_num - 1:
                    layer1 = nn.Linear(hidden_dim, output_dim)
                else:
                    layer1 = nn.Linear(hidden_dim, hidden_dim)

                layer1.weight.data.mul_(1e-3)
                nn.init.constant_(layer1.bias.data, 0.)
                self.mlp.append(layer1)


    def forward(self, x):
        for layer_index in range(self.layer_num - 1):
            layer1 = self.mlp[layer_index]
            # x = torch.relu(layer1(x))
            if self.activation == None:
                x = layer1(x)
                x = self.dropout(x) # remove dropout for the next version
            else:
                x = layer1(x)
                # x = self.dropout(x)
                x = self.activation(x)

        layer_lst = self.mlp[-1]

        return layer_lst(x)


class Agent:

    def __init__(self, problem, hyperparam, logger):

        self.problem = problem
        self.body_hidden_dim = hyperparam["body_hidden_dim"]
        self.body_layer_num = hyperparam["body_layer_num"]
        self.head_hidden_dim = hyperparam["head_hidden_dim"]
        self.head_layer_num = hyperparam["head_layer_num"]
        self.adv_hidden_dim = hyperparam["adv_hidden_dim"]
        self.adv_layer_num = hyperparam["adv_layer_num"]
        self.embedding_dim = hyperparam["embedding_dim"]
        self.test_size = hyperparam["problem_options"]["test_size"]
        self.val_size = hyperparam["problem_options"]["val_size"]
        self.debias_weight = hyperparam["debias_weight"]
        self.logger = logger
        self.root_log_dir = hyperparam["root_log_dir"]
        self.round_num = hyperparam["round_num"]
        self.learn_step = hyperparam["debias_step"]
        self.clf_body_learn_rate = hyperparam["clf_body_learn_rate"]
        self.clf_head_learn_rate = hyperparam["clf_head_learn_rate"]
        self.adv_learn_rate = hyperparam["adv_learn_rate"]
        self.pretrain_step = hyperparam["pretrain_step"]
        self.obj_acc = hyperparam["obj_acc"]


    def load_data(self, run_num):
        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)
        self.run_num = run_num
        self.output_dir = os.path.join(self.root_log_dir, "round" + str(self.run_num))

    def clf_body_init(self):


        self.clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                            activation=torch.relu)
        self.best_clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                            activation=torch.relu)

        # optimizer = torch.optim.RMSprop([weight_x, bias_x], lr=1e-3)
        self.optimizer_body = torch.optim.Adam(self.clf_body.parameters(), lr=self.clf_body_learn_rate)
        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)

    def clf_head_init(self):
        self.clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                            activation=torch.relu)
        self.best_clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                            activation=torch.relu)

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train, self.z_train), batch_size=256, shuffle=True, drop_last=True)

        self.optimizer_head = torch.optim.Adam(self.clf_head.parameters(), lr=self.clf_head_learn_rate)
        self.CE_criterion = torch.nn.CrossEntropyLoss()

    def clf_adv_init(self):

        self.adv = mlp(input_dim=self.embedding_dim, output_dim=2,
                        layer_num=self.adv_layer_num, hidden_dim=self.adv_hidden_dim,
                        activation=torch.relu)

        self.optimizer_adv = torch.optim.Adam(self.clf_head.parameters(), lr=self.adv_learn_rate)

    def adv_test(self):

        with torch.no_grad():

            x_head = self.clf_body(self.x_train)
            z_hat = self.adv(x_head).argmax(dim=1)
            z_acc = (z_hat == self.z_train).sum()/len(self.z_train)
            print("Z_ACC: {}".format(z_acc))



    def fit(self, run_num):

        self.clf_body_init()
        self.clf_head_init()
        self.clf_adv_init()

        for step in range(self.pretrain_step):
            self.clf_head.train()
            self.clf_body.train()
            for x, y, z in self.train_loader:
                self.clf_step(x, y)


            test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                              self.clf_head, None, None)

            self.logger.info('Vanilla Bias')
            self.logger.info("Acc = %s,", test_result['accuracy'])
            self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'],
                             test_result['1_fpr'])
            self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'],
                             test_result['1_fnr'])
            self.logger.info("EO = %s,",
                             -abs(test_result['1_fnr'] - test_result['0_fnr']) - abs(
                                 test_result['0_fpr'] - test_result['1_fpr']))
            self.logger.info("Obj = %s,", test_result['objective'])
            self.logger.info("----------------------------")



        for step in range(self.pretrain_step):
            self.adv.train()
            for x, y, z in self.train_loader:
                self.adv_step(x, z)

        target_network_update(self.clf_body, self.best_clf_body)
        target_network_update(self.clf_head, self.best_clf_head)

        self.optimizer_body = torch.optim.Adam(self.clf_body.parameters(), lr=self.clf_body_learn_rate)
        self.optimizer_head = torch.optim.Adam(self.clf_head.parameters(), lr=self.clf_head_learn_rate)
        self.optimizer_adv = torch.optim.Adam(self.clf_head.parameters(), lr=self.adv_learn_rate)

        best_obj = 0
        for step in range(self.learn_step):
            self.clf_head.train()
            self.clf_body.train()
            self.adv.train()
            for x, y, z in self.train_loader:

                self.clf_debias(x, y, z, self.debias_weight) # [run_num])
                self.adv_step(x, z)

            test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                              self.clf_head, None, None)

            cur_EO = abs(test_result['1_fnr'] - test_result['0_fnr']) + abs(test_result['0_fpr'] - test_result['1_fpr'])
            cur_acc = test_result['accuracy']
            cur_obj = self.obj_acc * cur_acc - cur_EO
            if cur_obj > best_obj:
                target_network_update(self.clf_body, self.best_clf_body)
                target_network_update(self.clf_head, self.best_clf_head)
                best_obj = cur_obj

        best_test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val,
                                                               self.best_clf_body, self.best_clf_head, None, None)

        self.logger.info('ADV Debias')
        self.logger.info("Step %s Test result:", step)
        self.logger.info("Acc = %s,", best_test_result['accuracy'])
        self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", best_test_result['fpr_diff'], best_test_result['0_fpr'],
                         best_test_result['1_fpr'])
        self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", best_test_result['fnr_diff'], best_test_result['0_fnr'],
                         best_test_result['1_fnr'])
        self.logger.info("EO = %s,",
                         -abs(best_test_result['1_fnr'] - best_test_result['0_fnr']) - abs(best_test_result['0_fpr'] - best_test_result['1_fpr']))
        self.logger.info("Obj = %s,", best_test_result['objective'])
        self.logger.info("----------------------------")

        #

        save_checkpoint(os.path.join(self.output_dir, "clf_body", "clf_body.pth.tar"),
                        round_index=self.run_num,
                        clf_body_state_dict=self.best_clf_body.state_dict(),
                        layer_num=self.body_layer_num,
                        clf_input_dim=self.x_train.shape[1],
                        hidden_dim=self.body_hidden_dim,
                        clf_output_dim=self.embedding_dim,
                        )

        save_checkpoint(os.path.join(self.output_dir, "clf_head", "clf_head.pth.tar"),
                        round_index=self.run_num,
                        clf_head_state_dict=self.best_clf_head.state_dict(),
                        layer_num=self.head_layer_num,
                        clf_input_dim=self.embedding_dim,
                        hidden_dim=self.head_hidden_dim,
                        clf_output_dim=2,
                        )

        # save_checkpoint(os.path.join(self.output_dir, "adv", "adv.pth.tar"),
        #                 round_index=self.run_num,
        #                 adv_state_dict=self.adv.state_dict(),
        #                 layer_num=self.adv_layer_num,
        #                 clf_input_dim=self.embedding_dim,
        #                 hidden_dim=self.adv_hidden_dim,
        #                 clf_output_dim=2,
        #                 )

    def adv_step(self, x, z):

        x_head = self.clf_body(x).detach()
        z_hat = self.adv(x_head)

        adv_loss = self.CE_criterion(z_hat, z.type(torch.long))
        # print(adv_loss)

        self.optimizer_adv.zero_grad()
        adv_loss.backward()
        nn.utils.clip_grad_norm_(self.adv.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        self.optimizer_adv.step()

    def clf_step(self, x, y):

        x_head = self.clf_body(x)
        y_hat = self.clf_head(x_head)

        clf_loss = self.CE_criterion(y_hat, y.type(torch.long))

        # print(clf_loss)
        loss_value = clf_loss
        self.optimizer_body.zero_grad()
        self.optimizer_head.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(self.clf_body.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        nn.utils.clip_grad_norm_(self.clf_head.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        self.optimizer_body.step()
        self.optimizer_head.step()

    def clf_debias(self, x, y, z, debias_weight):

        x_head = self.clf_body(x)
        y_hat = self.clf_head(x_head)
        z_hat = self.adv(x_head)

        clf_loss = self.CE_criterion(y_hat, y.type(torch.long))
        adv_loss = self.CE_criterion(z_hat, z.type(torch.long))

        # print(clf_loss, adv_loss)

        loss_value = clf_loss - debias_weight * adv_loss
        self.optimizer_body.zero_grad()
        self.optimizer_head.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(self.clf_body.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        nn.utils.clip_grad_norm_(self.clf_head.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        self.optimizer_body.step()
        self.optimizer_head.step()


def target_network_update(source_net, target_net):
    for target_params, source_params in zip(target_net.parameters(), source_net.parameters()):
        target_params.data.copy_(source_params)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def log_init(hyperparam):
    checkpoint_name = hyperparam["checkpoint_name"]
    round_num = hyperparam["round_num"]

    import log_utils, time, glob, logging, sys

    fdir0 = os.path.join(hyperparam["log_rootdir"], checkpoint_name)
    log_utils.create_exp_dir(fdir0)

    fdir1 = os.path.join(fdir0, checkpoint_name + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    log_utils.create_exp_dir(fdir1, scripts_to_save=glob.glob('*.py'))

    for round_index in range(round_num):
        fdir2 = os.path.join(fdir1, "round" + str(round_index))
        log_utils.create_exp_dir(fdir2)
        fdir3 = os.path.join(fdir2, "clf_body")
        log_utils.create_exp_dir(fdir3)
        fdir3 = os.path.join(fdir2, "clf_head")
        log_utils.create_exp_dir(fdir3)
        fdir3 = os.path.join(fdir2, "adv")
        log_utils.create_exp_dir(fdir3)

    hyperparam["root_log_dir"] = fdir1

    logger = log_utils.get_logger(tag=(""), log_level=logging.INFO)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(fdir1, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("hyperparam = %s", hyperparam)

    return logger


    # msglogger = logging.getLogger()
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(fdir1, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)
    #
    # msglogger.info("hyperparam = %s", hyperparam)
    #
    # return msglogger

import argparse
parser = argparse.ArgumentParser("PSS")
# parser.add_argument('--dataset', type=str, default='adult', help='.')
parser.add_argument('--hyper', type=str, default='bank_adv.json', help='.')
args = parser.parse_args()

def main():

    import json
    json_fname = args.hyper
    with open(os.path.join("hyperparameters/adv", json_fname)) as json_file:
        hyperparam = json.load(json_file)

    # # hyperparam = dict()
    # hyperparam["seed"] = 0
    # # hyperparam["clf_body_hidden_dim"] = 64
    # # hyperparam["clf_body_layer_num"] = 3
    # # hyperparam["clf_head_hidden_dim"] = 64
    # # hyperparam["clf_head_layer_num"] = 1
    # hyperparam["adv_hidden_dim"] = 64
    # hyperparam["adv_layer_num"] = 1
    # # hyperparam["embedding_dim"] = 64
    # # hyperparam["test_size"] = 0.5
    # # hyperparam["val_size"] = 0.5
    # # hyperparam["debias_weight"] = 50 # [100, 20, 20, 100, 100]
    # hyperparam["log_rootdir"] = "log_adv"
    # # hyperparam["round_num"] = 5
    # hyperparam["pretrain_step"] = 10
    # # hyperparam["debias_step"] = 50
    # # hyperparam["checkpoint_name"] = "checkpoint_" + json_fname[0:-5]
    # hyperparam["clf_body_learn_rate"] = 1e-3
    # hyperparam["clf_head_learn_rate"] = 1e-3
    # hyperparam["adv_learn_rate"] = 1e-3
    # hyperparam["obj_acc"] = 1 # 2

    # debias_weight
    # adult 1, 10, 100


    setup_seed(hyperparam["seed"])
    logger = log_init(hyperparam)

    # save hyperparameters
    with open(os.path.join(hyperparam["root_log_dir"], json_fname), "w") as json_file:
        json.dump(hyperparam, json_file)

    problem = load_problem_from_options(hyperparam["problem_options"])

    for run_num in range(hyperparam["round_num"]):
        agent = Agent(problem, hyperparam, logger)
        agent.load_data(run_num=run_num)
        agent.fit(run_num=run_num)
        # test_measurment_buf.append(test_measurment)


if __name__ == "__main__":

    main()
