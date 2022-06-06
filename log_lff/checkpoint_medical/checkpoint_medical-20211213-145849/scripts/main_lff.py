import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

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
            # layer1.weight.data.mul_(1e-3)
            # nn.init.constant_(layer1.bias.data, 0.)
            self.mlp.append(layer1)

        else:
            for layer_index in range(layer_num):
                if layer_index == 0:
                    layer1 = nn.Linear(input_dim, hidden_dim)
                elif layer_index == layer_num - 1:
                    layer1 = nn.Linear(hidden_dim, output_dim)
                else:
                    layer1 = nn.Linear(hidden_dim, hidden_dim)

                # layer1.weight.data.mul_(1e-3)
                # nn.init.constant_(layer1.bias.data, 0.)
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
        # self.adv_hidden_dim = hyperparam["adv_hidden_dim"]
        # self.adv_layer_num = hyperparam["adv_layer_num"]
        self.embedding_dim = hyperparam["embedding_dim"]
        self.test_size = hyperparam["problem_options"]["test_size"]
        self.val_size = hyperparam["problem_options"]["val_size"]
        self.debias_weight = hyperparam["debias_weight"]
        self.logger = logger
        self.root_log_dir = hyperparam["root_log_dir"]
        self.round_num = hyperparam["round_num"]
        self.learn_step = hyperparam["debias_step"]
        self.clf_B_learn_rate = hyperparam["clf_B_learn_rate"]
        self.clf_D_learn_rate = hyperparam["clf_D_learn_rate"]
        self.obj_acc = hyperparam["obj_acc"]


    def load_data(self, run_num):
        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)
        self.run_num = run_num
        self.output_dir = os.path.join(self.root_log_dir, "round" + str(self.run_num))

    def clf_body_init(self):


        self.clf_body_B = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                            activation=torch.relu)
        self.clf_body_D = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                              layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                              activation=torch.relu)

        self.best_clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                            activation=torch.relu)

        # optimizer = torch.optim.RMSprop([weight_x, bias_x], lr=1e-3)
        self.optimizer_body_B = torch.optim.Adam(self.clf_body_B.parameters(), lr=self.clf_B_learn_rate)
        self.optimizer_body_D = torch.optim.Adam(self.clf_body_D.parameters(), lr=self.clf_D_learn_rate)
        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)

    def clf_head_init(self):
        self.clf_head_B = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                            activation=torch.relu)
        self.clf_head_D = mlp(input_dim=self.embedding_dim, output_dim=2,
                              layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                              activation=torch.relu)

        self.best_clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                            activation=torch.relu)

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train, self.z_train), batch_size=256, shuffle=True, drop_last=True)

        self.optimizer_head_B = torch.optim.Adam(self.clf_head_B.parameters(), lr=self.clf_B_learn_rate)
        self.optimizer_head_D = torch.optim.Adam(self.clf_head_D.parameters(), lr=self.clf_D_learn_rate)
        self.CE_criterion = torch.nn.CrossEntropyLoss()

    # def clf_adv_init(self):
    #
    #     self.adv = mlp(input_dim=self.embedding_dim, output_dim=2,
    #                     layer_num=self.adv_layer_num, hidden_dim=self.adv_hidden_dim,
    #                     activation=torch.relu)
    #
    #     self.optimizer_adv = torch.optim.Adam(self.clf_head.parameters(), lr=1e-3)
    #
    # def adv_test(self):
    #
    #     with torch.no_grad():
    #
    #         x_head = self.clf_body(self.x_train)
    #         z_hat = self.adv(x_head).argmax(dim=1)
    #         z_acc = (z_hat == self.z_train).sum()/len(self.z_train)
    #         print("Z_ACC: {}".format(z_acc))



    def fit(self, run_num):

        self.clf_body_init()
        self.clf_head_init()
        self.clf_B_optimizer = torch.optim.Adam([{'params': self.clf_head_B.parameters()},
                                                 {'params': self.clf_body_B.parameters()}], lr=self.clf_B_learn_rate)
        self.clf_D_optimizer = torch.optim.Adam([{'params': self.clf_head_D.parameters()},
                                                 {'params': self.clf_body_D.parameters()}], lr=self.clf_D_learn_rate)

        best_obj = 0
        for step in range(self.learn_step):
            self.clf_head_B.train()
            self.clf_body_B.train()
            self.clf_head_D.train()
            self.clf_body_D.train()
            for x, y, z in self.train_loader:

                self.clf_B_step(x, y, z, self.debias_weight)
                self.clf_D_step(x, y, z, self.debias_weight)

            test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body_D,
                                                              self.clf_head_D, None, None)

            # self.logger.info('LFF Debias')
            # self.logger.info("Step %s Test result:", step)
            # self.logger.info("Acc = %s,", test_result['accuracy'])
            # self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'],
            #                  test_result['1_fpr'])
            # self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'],
            #                  test_result['1_fnr'])
            # self.logger.info("EO = %s,",
            #                  -abs(test_result['1_fnr'] - test_result['0_fnr']) - abs(test_result['0_fpr'] - test_result['1_fpr']))
            # self.logger.info("Obj = %s,", test_result['objective'])
            # self.logger.info("----------------------------")

            cur_EO = abs(test_result['1_fnr'] - test_result['0_fnr']) + abs(test_result['0_fpr'] - test_result['1_fpr'])
            cur_acc = test_result['accuracy']

            # print(cur_acc, cur_EO)

            if test_result['1_fnr'] >= 1. or test_result['0_fnr'] >= 1.:
                continue

            cur_obj = self.obj_acc * cur_acc - cur_EO
            if cur_obj > best_obj:
                target_network_update(self.clf_body_D, self.best_clf_body)
                target_network_update(self.clf_head_D, self.best_clf_head)
                best_obj = cur_obj

        best_test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val,
                                                               self.best_clf_body, self.best_clf_head, None, None)

        self.logger.info('LFF Debias')
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


    def clf_B_step(self, x, y, z, debias_weight):

        x_head_B = self.clf_body_B(x)
        y_hat_B = self.clf_head_B(x_head_B)
        y_hat_B_pro = F.softmax(y_hat_B, dim=1)
        # y_long = y.type(torch.long)
        y_hat_B_pro_y = y_hat_B_pro.T[0] * (1 - y) + y_hat_B_pro.T[1] * y

        clf_B_loss = torch.mean((debias_weight)**(-1) * (1-y_hat_B_pro_y)**(debias_weight)) # self.CE_criterion(y_hat, y.type(torch.long))
        # print(clf_B_loss.shape)
        # print(y_hat_B[0])
        # print(y_hat_B_pro[0])
        # print("Loss B:", clf_B_loss)
        # # print(CE_y_hat_D)
        # print("============")

        # self.optimizer_body_B.zero_grad()
        self.clf_B_optimizer.zero_grad()
        clf_B_loss.backward()
        # nn.utils.clip_grad_norm_(self.clf_body_B.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        # nn.utils.clip_grad_norm_(self.clf_head_B.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        self.clf_B_optimizer.step()
        # self.optimizer_head_B.step()


    def clf_D_step(self, x, y, z, debias_weight):

        x_head_D = self.clf_body_D(x)
        y_hat_D  = self.clf_head_D(x_head_D)
        y_hat_D_log_pro = -F.log_softmax(y_hat_D, dim=1)

        with torch.no_grad():
            x_head_B = self.clf_body_B(x)
            y_hat_B = self.clf_head_B(x_head_B)
            y_hat_B_log_pro = -F.log_softmax(y_hat_B, dim=1)
            # y_long = y.type(torch.long)
            CE_y_hat_B = y_hat_B_log_pro.T[0] * (1-y) + y_hat_B_log_pro.T[1] * y

        CE_y_hat_D = y_hat_D_log_pro.T[0] * (1-y) + y_hat_D_log_pro.T[1] * y
        weight = CE_y_hat_B/(CE_y_hat_B + CE_y_hat_D)
        # weight /= weight.sum()

        # # print(weight)
        # print(y_hat_B[0])
        # print(y_hat_B_pro[0])
        # print(CE_y_hat_B)
        # # print(CE_y_hat_D)
        # print("============")
        # print(weight)

        clf_D_loss = torch.matmul(weight.T, CE_y_hat_D)
        # clf_D_loss = CE_y_hat_D.mean()
        # clf_D_loss = self.CE_criterion(y_hat_D, y.type(torch.long))
        # print("Loss D:", clf_D_loss)

        self.clf_D_optimizer.zero_grad()
        # self.optimizer_head_D.zero_grad()
        clf_D_loss.backward()
        # nn.utils.clip_grad_norm_(self.clf_body_D.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        # nn.utils.clip_grad_norm_(self.clf_head_D.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
        self.clf_D_optimizer.step()
        # self.optimizer_head_D.step()


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
parser.add_argument('--hyper', type=str, default='medical_lff.json', help='.')
args = parser.parse_args()

def main():

    import json
    json_fname = args.hyper
    with open(os.path.join("hyperparameters/lff", json_fname)) as json_file:
        hyperparam = json.load(json_file)

    # # hyperparam = dict()
    # hyperparam["seed"] = 0
    # # hyperparam["clf_body_hidden_dim"] = 64
    # # hyperparam["clf_body_layer_num"] = 3
    # # hyperparam["clf_head_hidden_dim"] = 64
    # # hyperparam["clf_head_layer_num"] = 1
    # # hyperparam["adv_hidden_dim"] = 64
    # # hyperparam["adv_layer_num"] = 1
    # # hyperparam["embedding_dim"] = 64
    # # hyperparam["test_size"] = 0.5
    # # hyperparam["val_size"] = 0.5
    # # hyperparam["debias_weight"] = 50 # [100, 20, 20, 100, 100]
    # hyperparam["log_rootdir"] = "log_lff"
    # # hyperparam["round_num"] = 5
    # hyperparam["pretrain_step"] = 10
    # # hyperparam["debias_step"] = 50
    # # hyperparam["checkpoint_name"] = "checkpoint_" + json_fname[0:-5]
    # hyperparam["clf_body_learn_rate"] = 1e-3
    # hyperparam["clf_head_learn_rate"] = 1e-3
    # # hyperparam["adv_learn_rate"] = 1e-3
    # hyperparam["obj_acc"] = 1


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
