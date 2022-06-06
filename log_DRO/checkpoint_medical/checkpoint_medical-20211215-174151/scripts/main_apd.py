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
        self.embedding_dim = hyperparam["embedding_dim"]
        self.body_hidden_dim = hyperparam["body_hidden_dim"]
        self.body_layer_num = hyperparam["body_layer_num"]
        self.body_activation = hyperparam["body_activation"]
        self.body_fit_step = hyperparam["body_fit_step"]
        self.head_hidden_dim = hyperparam["head_hidden_dim"]
        self.head_layer_num = hyperparam["head_layer_num"]
        self.head_activation = hyperparam["head_activation"]
        self.learn_step = hyperparam["debias_step"]
        self.warmup_labels = hyperparam["warmup_labels"]
        self.debias_weight = hyperparam["debias_weight"]
        self.debias_gamma = hyperparam["debias_gamma"]
        self.learn_rate = hyperparam["learn_rate"]
        self.checkpoint_name = hyperparam["checkpoint_name"]
        self.round_num = hyperparam["round_num"]
        self.root_log_dir = hyperparam["root_log_dir"]
        self.logger = logger
        self.tune_fp = hyperparam["tune_fp"]
        self.tune_fn = hyperparam["tune_fn"]
        self.label_budget_percent = hyperparam["label_budget_percent"]


    def load_data(self, run_num):
        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)
        self.if_label = np.array([False] * self.x_train.shape[0])
        self.run_num = run_num
        self.output_dir = os.path.join(self.root_log_dir, "round" + str(self.run_num))
        self.label_budget = int(self.label_budget_percent * self.x_test.shape[0])

        index00 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 0]
        index01 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 1]
        index10 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 0]
        index11 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 1]
        self.logger.info("Training set 00 number %s, 01 number %s, 10 number %s, 11 number %s,",
                    len(index00), len(index01), len(index10), len(index11))
        print(self.problem.X.shape)

    def clf_z_init(self):
        self.clf_z = LogisticRegression()

    def clf_body_init(self):
        self.clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim, activation=eval(self.body_activation))

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train, self.z_train), batch_size=256, shuffle=True, drop_last=True)

        # optimizer = torch.optim.RMSprop([weight_x, bias_x], lr=1e-3)
        self.optimizer_body = torch.optim.Adam(self.clf_body.parameters(), lr=self.learn_rate)
        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)
        self.CE_criterion = torch.nn.CrossEntropyLoss()

    def clf_head_init(self):
        self.clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim, activation=eval(self.head_activation))

        # print(self.clf_body)
        # print(self.clf_head)
        self.optimizer_head = torch.optim.Adam(self.clf_head.parameters(), lr=self.learn_rate)
        # self.optimizer_head = torch.optim.SGD(self.clf_head.parameters(), lr=1e-4, momentum=0.9)
        # self.optimizer_head = torch.optim.RMSprop(self.clf_head.parameters(), lr=1e-3)


    def fit(self):

        self.clf_z_init()
        self.clf_body_init()
        self.clf_head_init()
        self.clf_body_fit(MAX_step=self.body_fit_step) # self.learn_step)

        checkpoint_body_fname = os.path.join(self.output_dir, "clf_body", "clf_body_0.pth.tar")
        save_checkpoint(checkpoint_body_fname,
                        round_index=self.run_num,
                        label_num=0,
                        clf_body_state_dict=self.clf_body.state_dict(),
                        layer_num=self.body_layer_num,
                        clf_input_dim=self.x_train.shape[1],
                        hidden_dim=self.body_hidden_dim,
                        clf_output_dim=self.embedding_dim,
                        )

        checkpoint_head_fname = os.path.join(self.output_dir, "clf_head", "clf_head_0.pth.tar")
        save_checkpoint(checkpoint_head_fname,
                        round_index=self.run_num,
                        label_num=0,
                        layer_num=self.head_layer_num,
                        clf_head_state_dict=self.clf_head.state_dict(),
                        clf_input_dim=self.embedding_dim,
                        hidden_dim=self.head_hidden_dim,
                        clf_output_dim=2,
                        )

        test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                          self.clf_head, self.problem, {'fn_weight': 0., 'fp_weight': 0., 'gamma': 0.})
        self.logger.info('VLN Model')
        self.logger.info("Test result:")
        self.logger.info("Acc = %s,", test_result['accuracy'])
        self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'], test_result['1_fpr'])
        self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'], test_result['1_fnr'])
        self.logger.info("EO = %s,", test_result['1_fnr'] - test_result['0_fnr'] + test_result['0_fpr'] - test_result['1_fpr'])
        self.logger.info("Obj = %s,", test_result['objective'])
        self.logger.info("----------------------------")


        self.warm_up()
        test_measurment_buf = []
        # for index in range(label_num):
        while self.if_label.sum() < self.label_budget:

            self.clf_z_init()
            if self.generate_sensitive_label():

                self.clf_head_init()
                test_result = self.clf_head_fit(MAX_step=self.learn_step, weight=self.debias_weight, gamma=self.debias_gamma)  # round 1 weight 0.02
                test_measurment_buf.append(test_result)

            # else:
            #     test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body, self.clf_head, self.problem,
            #                                                   {'fn_weight': 0., 'fp_weight': 0., 'gamma': 0.})
            #     test_measurment_buf.append(test_result)

            label_num = self.if_label.sum()

            checkpoint_fname = os.path.join(self.output_dir, "clf_head", "clf_head_" + str(label_num) + ".pth.tar")
            save_checkpoint(checkpoint_fname,
                            round_index=self.run_num,
                            label_num=label_num,
                            layer_num=self.head_layer_num,
                            clf_head_state_dict=self.clf_head.state_dict(),
                            clf_input_dim=self.embedding_dim,
                            hidden_dim=self.head_hidden_dim,
                            clf_output_dim=2,
                            )

            self.logger.info('Label Number = %s,', label_num)
            self.logger.info("Test result:")
            self.logger.info("Acc = %s,", test_result['accuracy'])
            self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'], test_result['1_fpr'])
            self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'], test_result['1_fnr'])
            self.logger.info("EO = %s,", test_result['1_fnr'] - test_result['0_fnr'] + test_result['0_fpr'] - test_result['1_fpr'])
            self.logger.info("Obj = %s,", test_result['objective'])
            self.logger.info("----------------------------")

        checkpointz_fname = os.path.join(self.output_dir, "clfz", "clfz_" + str(self.label_budget_percent) + ".pth.tar")
        save_checkpoint(checkpointz_fname,
                        clfz=self.clf_z,
                        round_index=self.run_num,
                        label_budget_percent=self.if_label.sum(),
                        clfz_input_dim=self.x_train.shape[1],
                        clfz_output_dim=2,
                        )

    def warm_up(self):
        index0_ = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0]
        index1_ = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1]
        label_index0_ = np.random.choice(index0_, self.warmup_labels)
        label_index1_ = np.random.choice(index1_, self.warmup_labels)
        self.if_label[label_index0_] = True
        self.if_label[label_index1_] = True


    def encoder_update(self):
        index0_ = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 0]
        index1_ = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 1]

        if len(index0_) >= self.warmup_labels and len(index1_) >= self.warmup_labels: # at least 2 samples for each class

            self.clf_z.fit(self.x_train[self.if_label], self.z_train[self.if_label])
            z_prob_train = self.clf_z.predict_proba(self.x_train)
            auc_z = metrics.roc_auc_score(self.z_train, z_prob_train[:, 1])
            self.logger.info('Z AUC = %s,', auc_z)

            return True
        else:

            return False

    def z_inference(self, x):
        z_prob = self.clf_z.predict_proba(x)
        z_prob0 = z_prob[:, 0]
        z_prob1 = z_prob[:, 1]
        z_pred = np.round(z_prob1).astype(np.int)
        # z_pred = (z_prob1_train > 0.6).astype(np.int)

        return z_pred, z_prob0, z_prob1, z_prob

    def y_inference(self, x):
        x_head = self.clf_body(x)
        y_prob = torch.softmax(self.clf_head(x_head), dim=1)
        y_prob0 = y_prob[:, 0]
        y_prob1 = y_prob[:, 1]
        y_pred = torch.round(y_prob1).type(torch.int)
        # auc_y = metrics.roc_auc_score(self.y_train, y_prob1)
        # acc_y = metrics.accuracy_score(self.y_train, y_pred, normalize=True)
        # self.logger.info('Y AUC = %s,', auc_y)
        # self.logger.info('Y ACC = %s,', acc_y)

        return y_pred, y_prob0, y_prob1, y_prob

    def false_rate_est_hard(self, y, y_pred, z):

        y_1, y_1_pred, y_0, y_0_pred = split_by_protected_value(y, y_pred, z)
        # y_1, y_1_pred, y_0, y_0_pred = split_by_protected_value(self.y_val, y_pred, self.z_val)
        fnr1 = (y_1_pred[y_1 == 1] == 0).sum() * 1. / (y_1 == 1).sum()
        fnr0 = (y_0_pred[y_0 == 1] == 0).sum() * 1. / (y_0 == 1).sum()
        fpr1 = (y_1_pred[y_1 == 0] == 1).sum() * 1. / (y_1 == 0).sum()
        fpr0 = (y_0_pred[y_0 == 0] == 1).sum() * 1. / (y_0 == 0).sum()

        return fnr0, fnr1, fpr0, fpr1

    def false_rate_est_soft(self, y, y_prob0, y_prob1, z_prob0, z_prob1):

        pos_buf, pos_index = get_positive_samples([y_prob0, y_prob1, z_prob0, z_prob1], y)
        neg_buf, neg_index = get_negative_samples([y_prob0, y_prob1, z_prob0, z_prob1], y)

        y_prob0_pos, y_prob1_pos, z_prob0_pos, z_prob1_pos = pos_buf
        y_prob0_neg, y_prob1_neg, z_prob0_neg, z_prob1_neg = neg_buf

        fpr1 = self.eval_est(y_prob1_neg, z_prob1_neg) # .T.sum(axis=0) / z_prob1_neg.sum()
        fpr0 = self.eval_est(y_prob1_neg, z_prob0_neg) # .T.sum(axis=0) / z_prob0_neg.sum()
        fnr1 = self.eval_est(y_prob0_pos, z_prob1_pos) # .T.sum(axis=0) / z_prob1_pos.sum()
        fnr0 = self.eval_est(y_prob0_pos, z_prob0_pos) # .T.sum(axis=0) / z_prob0_pos.sum()

        return fnr0, fnr1, fpr0, fpr1

    def eval_est(self, y_prob, z_prob):
        return (y_prob.T * z_prob).T.sum(axis=0) / z_prob.sum()

    def group_selection(self, fpr1, fpr0, fnr1, fnr0):
        fpr_gap = np.abs(fpr1 - fpr0)
        fnr_gap = np.abs(fnr1 - fnr0)

        if (fpr_gap > fnr_gap and self.tune_fp and self.tune_fn) or self.tune_fn == False:
            if fpr1 > fpr0:
                return 1,0 # z_group, y_group
            else:
                return 0,0

        elif (self.tune_fn and self.tune_fp) or self.tune_fp == False:
            if fnr1 > fnr0:
                return 1,1
            else:
                return 0,1

    def local_selection(self, unlabel_index, label_index):
        x_head = self.clf_body(self.x_train)
        x_unlabel = x_head[unlabel_index]
        x_label = x_head[label_index]
        pair_distance = euclidean_distances(x_unlabel, x_label)
        min_index = pair_distance.min(axis=1).argmax(axis=0)
        return unlabel_index[min_index]

    def generate_sensitive_label(self):

        if not self.encoder_update():
            unlabel_index = [index for index in range(self.x_train.shape[0]) if not self.if_label[index]]
            index = np.random.choice(unlabel_index, 1)
            self.if_label[index] = True
            return False

        with torch.no_grad():


            #### group selection ####
            z_pred, z_prob0, z_prob1, z_prob = self.z_inference(self.x_train) # self.x_val #
            y_pred, y_prob0, y_prob1, y_prob = self.y_inference(self.x_train) # torch.from_numpy(self.x_val).type(torch.float)) #

            # fnr0, fnr1, fpr0, fpr1 = self.false_rate_est_hard(self.y_train, y_pred, z_pred) # self.y_val
            fnr0, fnr1, fpr0, fpr1 = self.false_rate_est_soft(self.y_train, y_prob0, y_prob1, z_prob0, z_prob1) # self.y_val

            self.logger.info("fpr0 = %s, fpr1 = %s, ", fpr0.item(), fpr1.item())
            self.logger.info("fnr0 = %s, fnr1 = %s, ", fnr0.item(), fnr1.item())

            z_select, y_select = self.group_selection(fpr1, fpr0, fnr1, fnr0)
            self.logger.info("Select z = %d, y = %d, ", z_select, y_select)

            #### Instance selection ####
            label_index = [index for index in range(self.x_train.shape[0]) if self.if_label[index]]
            z_pred, z_prob0, z_prob1, z_prob = self.z_inference(self.x_train)
            index_subset = [index for index in range(self.x_train.shape[0]) if (not self.if_label[index]) and z_pred[index] == z_select and self.y_train[index] == y_select]
            # index_subset = [index for index in range(self.x_train.shape[0]) if (not self.if_label[index]) and self.z_train[index] == z_select and self.y_train[index] == y_select]

            index = self.local_selection(index_subset, label_index)
            self.if_label[index] = True

            index00 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 0 and self.y_train[index] == 0]
            index01 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 0 and self.y_train[index] == 1]
            index10 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 1 and self.y_train[index] == 0]
            index11 = [index for index in range(self.x_train.shape[0]) if self.if_label[index] and self.z_train[index] == 1 and self.y_train[index] == 1]

            self.logger.info("Current 00 number %s, 01 number %s, 10 number %s, 11 number %s,", len(index00), len(index01), len(index10), len(index11))

            if (len(index00) < 1 or len(index10) < 1) and (len(index01) < 1 or len(index11) < 1):
                return False
            else:
                return True


    def clf_body_fit(self, MAX_step=40):
        for step in range(MAX_step):
            self.clf_body.train()
            for x, y, z in self.train_loader:

                x_head = self.clf_body(x)
                y_hat = self.clf_head(x_head)
                loss_value = self.CE_criterion(y_hat, y.type(torch.long))

                # print(weight_x.grad, bias_x.grad)
                self.optimizer_body.zero_grad()
                self.optimizer_head.zero_grad()
                loss_value.backward()
                self.optimizer_body.step()
                self.optimizer_head.step()

        with torch.no_grad():
            x_head = self.clf_body(torch.from_numpy(self.x_val).type(torch.float))
            y_prob = torch.softmax(self.clf_head(x_head), dim=1)
            y_prob1 = y_prob[:, 1]
            y_pred = torch.round(y_prob1).type(torch.int)
            auc_y = metrics.roc_auc_score(self.y_val, y_prob1)
            acc_y = metrics.accuracy_score(self.y_val, y_pred, normalize=True)
            self.logger.info('Y AUC = %s,', auc_y)
            self.logger.info('Y ACC = %s,', acc_y)


    def clf_head_fit(self, MAX_step=40, weight=200., gamma=0.):

        x_head_label = self.clf_body(self.x_train[self.if_label]).detach()

        x_1, y_1, x_0, y_0 = split_by_protected_value(x_head_label, self.y_train[self.if_label], self.z_train[self.if_label])

        x_1_pos, _ = get_positive_examples(x_1, y_1)
        x_1_neg, _ = get_negative_examples(x_1, y_1)
        x_0_pos, _ = get_positive_examples(x_0, y_0)
        x_0_neg, _ = get_negative_examples(x_0, y_0)

        fn_weight = weight
        fp_weight = weight

        for step in range(MAX_step):
            self.clf_body.train()
            for x, y, z in self.train_loader:

                x_head = self.clf_body(x).detach()
                y_hat = self.clf_head(x_head)

                loss_value = self.CE_criterion(y_hat, y.type(torch.long))

                if x_1_pos.shape[0] > 0 and x_0_pos.shape[0] > 0 and self.tune_fn:
                    x_1_pos_head = self.clf_head(x_1_pos)
                    x_0_pos_head = self.clf_head(x_0_pos)
                    rfn = torch.square((x_1_pos_head[:, 1] - x_1_pos_head[:, 0]).mean() - (x_0_pos_head[:, 1] - x_0_pos_head[:, 0]).mean())
                    loss_value += fn_weight * rfn

                if x_1_neg.shape[0] > 0 and x_0_neg.shape[0] > 0 and self.tune_fp:
                    x_1_neg_head = self.clf_head(x_1_neg)
                    x_0_neg_head = self.clf_head(x_0_neg)
                    rfp = torch.square((x_1_neg_head[:, 1] - x_1_neg_head[:, 0]).mean() - (x_0_neg_head[:, 1] - x_0_neg_head[:, 0]).mean())
                    loss_value += fp_weight * rfp

                # loss_value = self.CE_criterion(torch.matmul(x_head, self.weight_x) + self.bias_x, y.type(torch.long))

                self.optimizer_head.zero_grad()
                loss_value.backward()
                nn.utils.clip_grad_norm_(self.clf_head.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
                self.optimizer_head.step()

        test_real_measures = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body, self.clf_head, self.problem,
                                                            {'fn_weight': weight, 'fp_weight': weight, 'gamma': gamma})

        return test_real_measures

    


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
        fdir3 = os.path.join(fdir2, "clfz")
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
parser.add_argument('--hyper', type=str, default='ucla_law_race_preprocess.json', help='.')
args = parser.parse_args()

def main():

    # load hyperparameters
    json_fname = args.hyper

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
    hyperparam["log_rootdir"] = "log"

    setup_seed(hyperparam["seed"])
    logger = log_init(hyperparam)

    # save hyperparameters
    with open(os.path.join(hyperparam["root_log_dir"], json_fname), "w") as json_file:
        json.dump(hyperparam, json_file)

    problem = load_problem_from_options(hyperparam["problem_options"])

    for run_num in range(hyperparam["round_num"]):
        agent = Agent(problem, hyperparam, logger)
        agent.load_data(run_num=run_num)
        agent.fit()
        # test_measurment_buf.append(test_measurment)


if __name__ == "__main__":

    main()
