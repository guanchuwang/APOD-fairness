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
from utils import load_data, load_checkpoint, save_checkpoint

import random
from sklearn.metrics.pairwise import euclidean_distances
import os, shutil

class mlp(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.layer1.weight.data.mul_(1e-3)
        nn.init.constant_(self.layer1.bias.data, 0.)

    def forward(self, x):
        return self.layer1(x)

    # def predict_proba(self, x):
    #     return torch.softmax(self.forward(x), dim=1)


class Base_agent:

    def __init__(self, problem, hyperparam, logger):
        self.problem = problem
        self.embedding_dim = hyperparam["embedding_dim"]
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

    def load_data(self, run_num):
        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)
        self.if_label = np.array([False] * self.x_train.shape[0])
        self.run_num = run_num
        self.output_dir = os.path.join(self.root_log_dir, "round" + str(self.run_num))

        index00 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 0]
        index01 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 1]
        index10 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 0]
        index11 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 1]
        self.logger.info("Training set 00 number %s, 01 number %s, 10 number %s, 11 number %s,",
                    len(index00), len(index01), len(index10), len(index11))
        print(self.x_train.shape, self.y_train.shape, self.z_train.shape)

    def clf_z_init(self):
        self.clf_z = LogisticRegression()

    def clf_init(self):
        self.clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim)

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train, self.z_train),
                                       batch_size=256, shuffle=True, drop_last=True)
        self.weight_x = Variable(torch.randn(self.embedding_dim, 2) * 1e-3, requires_grad=True)
        self.bias_x = Variable(torch.randn(1, 2) * 1e-3, requires_grad=True)

        # print(self.weight_x)
        # print(self.bias_x)

        # optimizer = torch.optim.RMSprop([weight_x, bias_x], lr=1e-3)
        self.optimizer = torch.optim.Adam([{'params': self.clf_body.parameters()},
                                      {'params': self.weight_x},
                                      {'params': self.bias_x}], lr=self.learn_rate)
        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)
        self.CE_criterion = torch.nn.CrossEntropyLoss()

    def fit(self, label_budget):

        self.clf_z_init()
        self.clf_init()
        self.warm_up()
        self.unfair_fit(MAX_step=self.learn_step)

        checkpointv_fname = os.path.join(self.output_dir, "clf", "clf_0.pth.tar")
        save_checkpoint(checkpointv_fname,
                        clf_state_dict=self.clf_body.state_dict(),
                        weight=self.weight_x.detach(),
                        bias=self.bias_x.detach(),
                        round_index=self.run_num,
                        label_num=0,
                        clf_input_dim=self.x_train.shape[1],
                        clf_output_dim=self.embedding_dim,
                        )

        test_measurment_buf = []
        # for index in range(label_num):
        while self.if_label.sum() < label_budget:

            self.clf_z_init()
            if self.generate_sensitive_label():

                self.clf_init()
                test_result = self.fair_fit(MAX_step=self.learn_step, weight=self.debias_weight, gamma=self.debias_gamma)  # round 1 weight 0.02
                test_measurment_buf.append(test_result)

            else:
                test_result = measure_objective_results_torch(self.x_val, self.y_val, self.z_val,
                                                              self.clf_body, self.weight_x.detach(),
                                                              self.bias_x.detach(), self.problem,
                                                              {'fn_weight': 0., 'fp_weight': 0., 'gamma': 0.})
                test_measurment_buf.append(test_result)

            label_num = self.if_label.sum()

            checkpoint_fname = os.path.join(self.output_dir, "clf", "clf_" + str(label_num) + ".pth.tar")
            save_checkpoint(checkpoint_fname,
                            clf_state_dict=self.clf_body.state_dict(),
                            weight=self.weight_x.detach(),
                            bias=self.bias_x.detach(),
                            round_index=self.run_num,
                            label_num=label_num,
                            clf_input_dim=self.x_train.shape[1],
                            clf_output_dim=self.embedding_dim,
                            )

            self.logger.info('Label Number = %s,', label_num)
            self.logger.info("Test result:")
            self.logger.info("Acc = %s,", test_result['accuracy'])
            self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'], test_result['1_fpr'])
            self.logger.info("Fnr = %s, (Fpr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'], test_result['1_fnr'])
            self.logger.info("Obj = %s,", test_result['objective'])
            self.logger.info("----------------------------")

        checkpointz_fname = os.path.join(self.output_dir, "clfz", "clfz_" + str(label_budget) + ".pth.tar")
        save_checkpoint(checkpointz_fname,
                        clfz=self.clf_z,
                        round_index=self.run_num,
                        label_num=label_budget,
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

    def unfair_fit(self, MAX_step=40):
        for step in range(MAX_step):
            self.clf_body.train()
            for x, y, z in self.train_loader:

                x_head = self.clf_body(x)
                loss_value = self.CE_criterion(torch.matmul(x_head, self.weight_x) + self.bias_x, y.type(torch.long))

                # print(weight_x.grad, bias_x.grad)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

        with torch.no_grad():
            x_head = self.clf_body(torch.from_numpy(self.x_val).type(torch.float))
            y_prob = torch.softmax(torch.matmul(x_head, self.weight_x) + self.bias_x, dim=1)
            y_prob1 = y_prob[:, 1]
            y_pred = torch.round(y_prob1).type(torch.int)
            auc_y = metrics.roc_auc_score(self.y_val, y_prob1)
            acc_y = metrics.accuracy_score(self.y_val, y_pred, normalize=True)
            self.logger.info('Y AUC = %s,', auc_y)
            self.logger.info('Y ACC = %s,', acc_y)

    def generate_sensitive_label(self):

        pass


    def fair_fit(self):

        pass


class PUA_agent(Base_agent):

    def __init__(self, problem, hyperparam, logger):
        super().__init__(problem, hyperparam, logger)

    def fair_fit(self, MAX_step=40, weight=200., gamma=0.):

        for step in range(MAX_step):
            self.clf_body.train()
            for x, y, z in self.train_loader:

                x_head_label = self.clf_body(self.x_train[self.if_label])

                x_1, y_1, x_0, y_0 = split_by_protected_value(x_head_label, self.y_train[self.if_label], self.z_train[self.if_label])

                x_1_pos, _ = get_positive_examples(x_1, y_1)
                x_1_neg, _ = get_negative_examples(x_1, y_1)
                x_0_pos, _ = get_positive_examples(x_0, y_0)
                x_0_neg, _ = get_negative_examples(x_0, y_0)

                # print(x_0_neg.shape, x_0_pos.shape, x_1_neg.shape, x_1_pos.shape)

                fnr_diff, fpr_diff = 0, 0
                if x_1_pos.shape[0] > 0 and x_0_pos.shape[0] > 0 and self.tune_fn:
                    diff_pos = (torch.sum(x_1_pos, dim=0) / x_1_pos.shape[0]) - (torch.sum(x_0_pos, dim=0) / x_0_pos.shape[0])
                    fnr_diff = torch.square(torch.matmul(diff_pos, self.weight_x[:, 1] - self.weight_x[:, 0]))

                if x_1_neg.shape[0] > 0 and x_0_neg.shape[0] > 0 and self.tune_fp:
                    diff_neg = (torch.sum(x_1_neg, dim=0) / x_1_neg.shape[0]) - (torch.sum(x_0_neg, dim=0) / x_0_neg.shape[0])
                    fpr_diff = torch.square(torch.matmul(diff_neg, self.weight_x[:, 1] - self.weight_x[:, 0]))

                weight_norm_square = torch.square(self.weight_x).sum()
                bias_norm_square = torch.square(self.bias_x).sum()

                fn_weight = weight
                fp_weight = weight


                x_head = self.clf_body(x)

                loss_value = self.CE_criterion(torch.matmul(x_head, self.weight_x) + self.bias_x, y.type(torch.long)) \
                             + fp_weight * fpr_diff + fn_weight * fnr_diff \
                             + gamma * weight_norm_square + gamma * bias_norm_square

                # loss_value = self.CE_criterion(torch.matmul(x_head, self.weight_x) + self.bias_x, y.type(torch.long))

                # print(weight_x.grad, bias_x.grad)

                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()

        test_real_measures = measure_objective_results_torch(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                            self.weight_x.detach(), self.bias_x.detach(), self.problem,
                                                            {'fn_weight': weight, 'fp_weight': weight, 'gamma': gamma})

        return test_real_measures



class Base_testagent:

    def __init__(self, problem, hyperparam):
        self.problem = problem
        self.embedding_dim = hyperparam["embedding_dim"]

    def load_data(self, run_num, checkpoint_dirname):

        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(self.problem, run_num)

        print(self.x_train.shape, self.x_val.shape, self.x_test.shape)

        cur_path = os.getcwd()
        self.cp_path_clf = os.path.join(cur_path, checkpoint_dirname, "round" + str(run_num), "clf")

    def load_clf(self, label_num):

        cp_name = "clf_" + str(label_num) + ".pth.tar"
        cp_fullname = os.path.join(self.cp_path_clf, cp_name)
        checkpoint = load_checkpoint(cp_fullname)

        self.weight = checkpoint["weight"]
        self.bias = checkpoint["bias"]

        self.clf_body = mlp(input_dim=checkpoint["clf_input_dim"], output_dim=checkpoint["clf_output_dim"])
        self.clf_body.load_state_dict(checkpoint["clf_state_dict"])

    def validate(self, hyperparam):

        pass


    def validate_vallina(self):
        self.load_clf(label_num=0)
        test_measures = measure_objective_results_torch(self.x_test, self.y_test, self.z_test, self.clf_body,
                                                        self.weight, self.bias, self.problem, None)
        return test_measures

    





