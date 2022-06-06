import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


import cvxpy as cp
import numpy as np

from utils import load_problem_from_options, measure_objective_results, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results2
from utils import load_data, save_checkpoint


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


def sample_selection(unlabel_index, label_index, x):
    x_unlabel = x[unlabel_index]
    x_label = x[label_index]
    pair_distance = euclidean_distances(x_unlabel, x_label)
    min_index = pair_distance.min(axis=1).argmax(axis=0)
    return unlabel_index[min_index]


class Agent:

    def __init__(self, problem):
        self.problem = problem
        self.embedding_dim = 20

    def load_data(self, run_num):
        self.x_train, self.x_val, self.x_test, \
        self.y_train, self.y_val, self.y_test, \
        self.z_train, self.z_val, self.z_test = load_data(problem, run_num)
        self.if_label = np.array([False] * self.x_train.shape[0])
        self.run_num = run_num
        self.output_dir = os.path.join("checkpoint_baseline_active", "round" + str(self.run_num))

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.mkdir(self.output_dir)
        os.mkdir(os.path.join(self.output_dir, "clf"))
        os.mkdir(os.path.join(self.output_dir, "clfz"))

        index00 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 0]
        index01 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 0 and self.y_train[index] == 1]
        index10 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 0]
        index11 = [index for index in range(self.x_train.shape[0]) if self.z_train[index] == 1 and self.y_train[index] == 1]
        print("Training set:", len(index00), len(index01), len(index10), len(index11))

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
                                      {'params': self.bias_x}], lr=1e-3)
        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)
        self.CE_criterion = torch.nn.CrossEntropyLoss()

    def fit(self, label_num):

        self.clf_z_init()
        self.clf_init()
        self.unfair_fit(MAX_step=40)
        test_measurment_buf = []

        for index in range(label_num):
            print('Label Number:', self.if_label.sum())

            self.generate_sensitive_label()
            self.clf_z_init()
            self.clf_init()
            test_result = self.fair_fit(MAX_step=40, weight=0.5, gamma=0.)  # round 1 weight 0.02
            test_measurment_buf.append(test_result)

            checkpoint_fname = os.path.join(self.output_dir, "clf", "clf_" + str(index) + ".pth.tar")
            # "./checkpoint_active/clf/clf_" + str(self.run_num) + "_" + str(index) + ".pth.tar"
            save_checkpoint(checkpoint_fname,
                            clf_state_dict=self.clf_body.state_dict(),
                            weight=self.weight_x.detach(),
                            bias=self.bias_x.detach(),
                            round_index=self.run_num,
                            label_num=index,
                            clf_input_dim=self.x_train.shape[1],
                            clf_output_dim=self.embedding_dim,
                            )

            print("Test result:")
            print("Acc", test_result['accuracy'], ",")
            print("Fpr", test_result['fpr_diff'], ",")
            print("Fnr", test_result['fnr_diff'], ",")
            print("Obj", test_result['objective'], ",")
            print("----------------------------")

        self.clf_z.fit(self.x_train[self.if_label], self.z_train[self.if_label])
        checkpointz_fname = os.path.join(self.output_dir, "clfz", "clfz_" + str(label_num) + ".pth.tar")
        save_checkpoint(checkpointz_fname,
                        clfz=self.clf_z,
                        round_index=self.run_num,
                        label_num=label_num,
                        clfz_input_dim=self.x_train.shape[1],
                        clfz_output_dim=2,
                        )

    def generate_sensitive_label(self):

        label_index = [index for index in range(self.x_train.shape[0]) if self.if_label[index]]
        if len(label_index) > 0:

            with torch.no_grad():
                x_train_head = self.clf_body(self.x_train)
                unlabel_index = [index for index in range(self.x_train.shape[0]) if (not self.if_label[index])]
                index = sample_selection(unlabel_index, label_index, x_train_head)

        else:

                index = np.random.randint(0, self.x_train.shape[0], 1)

        self.if_label[index] = True

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

    def fair_fit(self, MAX_step=40, weight=200., gamma=0.):

    # gammas = np.linspace(problem.gamma_gt, problem.gamma_lt, num=problem.gamma_res)
    # weights = np.linspace(problem.weight_gt, problem.weight_lt, num=problem.weight_res)

    # for weight in weights:
    #     for gamma in gammas:
    #         print(weight, gamma)
    #   weight, gamma = 200., 0.

        for step in range(MAX_step):
            self.clf_body.train()
            for x, y, z in self.train_loader:

                fnr_diff, fpr_diff = 0, 0
                weight_norm_square, bias_norm_square = 0, 0
                if self.if_label.sum() > 0:

                    x_head_label = self.clf_body(self.x_train[self.if_label])

                    x_1, y_1, x_0, y_0 = split_by_protected_value(x_head_label, self.y_train[self.if_label], self.z_train[self.if_label])

                    x_1_pos, _ = get_positive_examples(x_1, y_1)
                    x_1_neg, _ = get_negative_examples(x_1, y_1)
                    x_0_pos, _ = get_positive_examples(x_0, y_0)
                    x_0_neg, _ = get_negative_examples(x_0, y_0)

                    # print(x_0_neg.shape, x_0_pos.shape, x_1_neg.shape, x_1_pos.shape)

                    if x_1_pos.shape[0] > 0 and x_0_pos.shape[0] > 0:
                        diff_pos = (torch.sum(x_1_pos, dim=0) / x_1_pos.shape[0]) - (torch.sum(x_0_pos, dim=0) / x_0_pos.shape[0])
                        fnr_diff = torch.square(torch.matmul(diff_pos, self.weight_x[:, 1] - self.weight_x[:, 0]))

                    if x_1_neg.shape[0] > 0 and x_0_neg.shape[0] > 0:
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

        # print(weight_x, bias_x)
        # print(type(self.x_train), type(self.y_train), type(self.z_train))

        # label_index = [index for index in range(self.x_train.shape[0]) if self.if_label[index]]


        # print(self.x_train.shape, self.y_train.shape, self.z_train.shape)

        test_real_measures = measure_objective_results2(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                        self.weight_x.detach(), self.bias_x.detach(), problem,
                                                        {'fn_weight': weight, 'fp_weight': weight, 'gamma': gamma})

        return test_real_measures


# z_prob_val = self.clf_z.predict_proba(self.x_val)
# test_label_measures = measure_objective_results3(self.x_val, self.y_val, z_prob_val, self.clf_body,
#                                                 self.weight_x.detach(), self.bias_x.detach(), problem,
#                                                 {'fn_weight': weight, 'fp_weight': weight, 'gamma': gamma})


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




# np_seed = 7
# torch_seed = 7
# np.random.seed(np_seed)
setup_seed(0)
problem = load_problem_from_options('./options/adult_2.json')
# problem = load_problem_from_options('./options/compas_.json')
# test_measurment_buf = []
label_num = 40
total_run_num = 5 # 5
# run_round = 2 # 0
for run_num in range(total_run_num):

    agent = Agent(problem)
    agent.load_data(run_num=run_num)
    agent.fit(label_num=label_num)

    # test_measurment_buf.append(test_measurment)

    # run_round += 1
    # if run_round >= run_num:
    #     break

    # print("Best test result:")
    # print(best_test_result['accuracy'])
    # print(best_test_result['fpr_diff'])
    # print(best_test_result['fnr_diff'])
    # print(best_test_result['objective'])
    # print("----------------------------")

# agent.load_data(run_num=0)
# agent.fit(label_num=label_num)

    # measure = train_fair_classifier(problem, run_num)
    # test_measurment.append(measure)

# mean_accuracy = np.array([[y['accuracy'] for y in x] for x in test_measurment_buf]).mean(axis=0)
# mean_objective = np.array([[y['objective'] for y in x] for x in test_measurment_buf]).mean(axis=0)
# mean_fpr_diff = np.array([[y['fpr_diff'] for y in x] for x in test_measurment_buf]).mean(axis=0)
# mean_fnr_diff = np.array([[y['fnr_diff'] for y in x] for x in test_measurment_buf]).mean(axis=0)
#
# print("Mean Best result:")
# print("Acc", mean_accuracy)
# print("Fpr", mean_fpr_diff)
# print("Fnr", mean_fnr_diff)
# print("Obj", mean_objective)
# print("----------------------------")


#####################################
# ACC too low: labels are not enough