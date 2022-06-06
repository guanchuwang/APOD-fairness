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

from utils import load_problem_from_options, measure_objective_results_nobody, split_by_protected_value
from utils import get_positive_examples, get_negative_examples, get_positive_samples, get_negative_samples
from utils import measure_objective_results_torch


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



class mlp(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer1.weight.data.mul_(1e-3)
        nn.init.constant_(self.layer1.bias.data, 0.)

        # self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # self.layer2.weight.data.mul_(1e-3)
        # nn.init.constant_(self.layer2.bias.data, 0.)

        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.layer3.weight.data.mul_(1e-3)
        nn.init.constant_(self.layer3.bias.data, 0.)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        # x = torch.relu(self.layer2(x))
        return self.layer3(x)



import os, json
json_fname = 'adult.json'
with open(os.path.join("hyperparameters", json_fname)) as json_file:
    hyperparam = json.load(json_file)

# np_seed = 0
# np.random.seed(np_seed)
setup_seed(0)
# problem = load_problem_from_options('./options/compas_.json')
# problem = load_problem_from_options('./options/adult_.json')
# problem = load_problem_from_options('./options/region_job.json')
# problem = load_problem_from_options('./options/ucla-law-school_protected-race.json')
problem = load_problem_from_options(hyperparam["problem_options"])
# print(problem.X.shape)
# print(problem.X[0])

Z = problem.X[:, problem.protected_index].astype(np.int)
X = np.concatenate((problem.X[:, :problem.protected_index], problem.X[:, problem.protected_index+1:]), axis=1)
X = StandardScaler().fit_transform(X)

x_train_all, x_test, y_train_all, y_test, z_train_all, z_test = train_test_split(X, problem.Y, Z,
                                                                                 test_size=problem.test_size,
                                                                                 random_state=0) # cross validation, random_state=run_num)

x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train_all, y_train_all, z_train_all,
                                                                  test_size=problem.val_size,
                                                                  random_state=0) # , random_state=fold)

y_train = y_train.astype(np.int)
y_val = y_val.astype(np.int)
y_test = y_test.astype(np.int)

print(x_train.shape, y_train.shape, z_train.shape)

# print(train_loader)
# label_size = 40
# label_index = np.random.choice(np.arange(0, x_train.shape[0]), label_size)

x_train = torch.from_numpy(x_train).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.int)
z_train = torch.from_numpy(z_train).type(torch.int)

# test_loader = DataLoader(TensorDataset(x_test, y_test, z_test), batch_size=256, shuffle=False, drop_last=False)

# clf_y = LogisticRegression()
# clf_y.fit(StandardScaler().fit_transform(x_train), y_train)
# y_prob = clf_y.predict_proba(StandardScaler().fit_transform(x_train))
# y_prob0 = y_prob[:, 0]
# y_prob1 = y_prob[:, 1]
# y_pred = (y_prob1 > 0.5).astype(np.int)
# auc_y = metrics.roc_auc_score(y_train, y_prob1)
# acc_y = metrics.accuracy_score(y_train, y_pred, normalize=True)
# print('Y AUC:', auc_y)
# print('Y ACC:', acc_y)

# gammas = np.linspace(problem.gamma_gt, problem.gamma_lt, num=problem.gamma_res)
# weights = np.linspace(problem.weight_gt, problem.weight_lt, num=problem.weight_res)
gammas = np.linspace(0, 5, num=1)
weights = np.linspace(20, 400, num=20)

test_measurment = []

for weight in weights:
    for gamma in gammas:
        print(weight, gamma)

        weight, gamma = 0.03, 0.,

        train_loader = DataLoader(TensorDataset(x_train, y_train, z_train), batch_size=256, shuffle=True, drop_last=True)
        # weight_x = Variable(torch.randn(x_train.shape[1], 2) * 1e-3, requires_grad=True)
        embedding_dim = 20

        weight_x = Variable(torch.randn(embedding_dim, 2) * 1e-3, requires_grad=True)
        bias_x = Variable(torch.randn(2) * 1e-3, requires_grad=True) # torch.tensor(0.) #
        clf_body = mlp(input_dim=x_train.shape[1], output_dim=embedding_dim, hidden_dim=64)

        # optimizer = torch.optim.RMSprop([weight_x, bias_x], lr=1e-3) # , bias_x]
        # optimizer = torch.optim.Adam([weight_x, bias_x], lr=1e-3)
        optimizer = torch.optim.Adam([{'params': clf_body.parameters()},
                                        {'params': weight_x},
                                        {'params': bias_x}], lr=1e-3)

        # optimizer = torch.optim.SGD([weight_x, bias_x], lr=1e-3, momentum=0.9)
        # optimizer = torch.optim.SGD([{'params': clf_body.parameters()},
        #                                         {'params': weight_x},
        #                                         {'params': bias_x}], lr=1e-4, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()


        for step in range(1, 40):

            for x, y, z in train_loader:

                # print(x_1_pos.shape, x_1_neg.shape, x_0_pos.shape, x_0_neg.shape)
                # log_likelihood = torch.sum(y_train * (torch.matmul(x_train, weight_x) + bias_x)) \
                #                     - torch.sum(torch.log(torch.sigmoid(torch.matmul(x_train, weight_x) + bias_x) + 1e-8))
                #
                # print(x.shape, weight_x.shape)

                # y_prob1 = torch.sigmoid(torch.matmul(x, weight_x))
                # log_likelihood = torch.mean(y * torch.log(y_prob1 + 1e-8) + (1-y) * torch.log(1 - y_prob1 + 1e-8))

                x_head = clf_body(x)

                # print(x_head)

                x_1, y_1, x_0, y_0 = split_by_protected_value(x_head, y.type(torch.long), z.type(torch.long))
                x_1_pos, _ = get_positive_examples(x_1, y_1)
                x_1_neg, _ = get_negative_examples(x_1, y_1)
                x_0_pos, _ = get_positive_examples(x_0, y_0)
                x_0_neg, _ = get_negative_examples(x_0, y_0)

                # print(x_1_pos.shape[0], x_1_neg.shape[0], x_0_pos.shape[0], x_0_neg.shape[0])

                diff_pos = (torch.sum(x_1_pos, dim=0) / x_1_pos.shape[0]) - (torch.sum(x_0_pos, dim=0) / x_0_pos.shape[0])
                diff_neg = (torch.sum(x_1_neg, dim=0) / x_1_neg.shape[0]) - (torch.sum(x_0_neg, dim=0) / x_0_neg.shape[0])

                # print(diff_pos, diff_neg)

                fpr_diff = torch.square(torch.matmul(diff_neg, weight_x[:, 0] - weight_x[:, 1]))
                fnr_diff = torch.square(torch.matmul(diff_pos, weight_x[:, 0] - weight_x[:, 1]))

                # print(fpr_diff, fnr_diff)

                weight_norm_square = torch.square(weight_x).sum()
                bias_norm_square = torch.square(bias_x).sum()

                fn_weight = weight
                fp_weight = weight

                # log_likelihood = torch.matmul(y.type(torch.float).T, torch.matmul(x, weight_x) + bias_x) - \
                #                     torch.log(1. + torch.exp(torch.matmul(x, weight_x) + bias_x)).sum()
                # loss_value = -log_likelihood + fn_weight * fpr_diff + fn_weight * fnr_diff \
                #              + gamma * weight_norm_square # + gamma * bias_norm_square
                #
                # loss_value = -log_likelihood

                # print(fn_weight * fpr_diff, fn_weight * fnr_diff)

                loss_value = criterion(torch.matmul(x_head, weight_x) + bias_x, y.type(torch.long)) \
                             + fn_weight * fpr_diff + fn_weight * fnr_diff \
                             + gamma * weight_norm_square + gamma * bias_norm_square


                # print(weight_x.grad, bias_x.grad)

                if torch.isnan(loss_value):
                    print("Loss: ", loss_value)


                optimizer.zero_grad()
                loss_value.backward()
                nn.utils.clip_grad_norm_(clf_body.parameters(), max_norm=0.1, norm_type=2)  # necessary!!!
                nn.utils.clip_grad_norm_(weight_x, max_norm=0.1, norm_type=2)  # necessary!!!
                nn.utils.clip_grad_norm_(bias_x, max_norm=0.1, norm_type=2)  # necessary!!!
                optimizer.step()

        # print(weight_x, bias_x)

        # y_pred_val = torch.softmax(torch.matmul(torch.from_numpy(x_val).type(torch.float), weight_x) + bias_x, axis=1)[:, 1]
        # auc_y = metrics.roc_auc_score(y_val, y_pred_val.detach().numpy())
        # # acc_y = metrics.accuracy_score(y_train, y_pred, normalize=True)
        # print('Y AUC:', auc_y)

        # y_hat = torch.round(torch.sigmoid(torch.matmul(x_train, weight_x)))  # + b))
        # y_hat_0 = torch.round(torch.sigmoid(torch.matmul(x_0, weight_x)))  # + b))
        # y_hat_1 = torch.round(torch.sigmoid(torch.matmul(x_1, weight_x)))  # + b))
        # y_hat_1_pos, _ = get_positive_examples(y_hat_1, y_1)
        # y_hat_1_neg, _ = get_negative_examples(y_hat_1, y_1)
        # y_hat_0_pos, _ = get_positive_examples(y_hat_0, y_0)
        # y_hat_0_neg, _ = get_negative_examples(y_hat_0, y_0)
        #
        # acc = (y_hat.squeeze(dim=1) == y_train).sum()*1./y_train.shape[0]
        # fnr0 = (y_hat_0_pos == 0).sum()*1./y_hat_0_pos.shape[0]
        # fnr1 = (y_hat_1_pos == 0).sum()*1./y_hat_1_pos.shape[0]
        # fpr0 = (y_hat_0_neg == 1).sum()*1./y_hat_0_neg.shape[0]
        # fpr1 = (y_hat_1_neg == 1).sum()*1./y_hat_1_neg.shape[0]
        #
        # fnr_gap = torch.abs(fnr0 - fnr1)
        # fpr_gap = torch.abs(fpr0 - fpr1)
        #
        # print("ACC:", acc)
        # print(fnr_diff, fpr_diff)
        # print(fnr_gap, fpr_gap)

        # test_real_measures = measure_objective_results_nobody(x_train.numpy(), y_train.numpy(), z_train.numpy(), weight_x.detach(), bias_x, problem,
        #                                                {'fn_weight': fn_weight, 'fp_weight': fp_weight, 'gamma': gamma})

        test_real_measures = measure_objective_results_torch(x_val, y_val, z_val, clf_body,
                                                             weight_x.detach(), bias_x.detach(), problem,
                                                             {'fn_weight': weight, 'fp_weight': weight, 'gamma': gamma})

        test_measurment.append(test_real_measures)

        print(test_real_measures['accuracy'])
        print(test_real_measures['fpr_diff'])
        print(test_real_measures['fnr_diff'])
        print(test_real_measures['objective'])
        print("----------------------------")
        #


objective_buf = np.array([measurment['objective'] for measurment in test_measurment])
max_objective_index = objective_buf.argmin()
max_test_measurment = test_measurment[max_objective_index]

print("------------------------------")
print(max_test_measurment['accuracy'])
print(max_test_measurment['fpr_diff'])
print(max_test_measurment['fnr_diff'])
print(max_test_measurment['objective'])
print(max_test_measurment['fp_weight'])
print(max_test_measurment['fn_weight'])
print(max_test_measurment['gamma'])