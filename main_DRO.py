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


class LossComputer:
    def __init__(self, criterion, is_robust, dataset, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset["n_groups"]
        self.group_counts = dataset["group_counts"]
        self.group_frac = dataset["group_frac"]
        self.group_str = ["0", "1"]

        if adj is not None:
            self.adj = torch.from_numpy(adj).float() 
        else:
            self.adj = torch.zeros(self.n_groups).float()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups)/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust:
            # actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups)
        self.update_data_counts = torch.zeros(self.n_groups)
        self.update_batch_counts = torch.zeros(self.n_groups)
        self.avg_group_loss = torch.zeros(self.n_groups)
        self.avg_group_acc = torch.zeros(self.n_groups)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str[group_idx]}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()


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
        self.learn_rate = hyperparam["learn_rate"]
        self.checkpoint_name = hyperparam["checkpoint_name"]
        self.round_num = hyperparam["round_num"]
        self.root_log_dir = hyperparam["root_log_dir"]
        self.logger = logger
        self.tune_fp = hyperparam["tune_fp"]
        self.tune_fn = hyperparam["tune_fn"]
        self.label_budget_percent = hyperparam["label_budget_percent"]
        self.debias_gamma = hyperparam["debias_gamma"]
        self.debias_alpha = hyperparam["debias_alpha"]
        self.obj_acc = hyperparam["obj_acc"]


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

        self.group_counts = torch.tensor([len(index00) + len(index01), len(index10) + len(index11)])
        self.group_frac = self.group_counts*1./self.group_counts.sum()

    def clf_init(self):

        self.clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                            layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                            activation=eval(self.body_activation))

        self.clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                            layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                            activation=eval(self.head_activation))

        self.best_clf_body = mlp(input_dim=self.x_train.shape[1], output_dim=self.embedding_dim,
                                 layer_num=self.body_layer_num, hidden_dim=self.body_hidden_dim,
                                 activation=torch.relu)

        self.best_clf_head = mlp(input_dim=self.embedding_dim, output_dim=2,
                                 layer_num=self.head_layer_num, hidden_dim=self.head_hidden_dim,
                                 activation=torch.relu)

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train.type(torch.long), self.z_train), batch_size=256,
                                       shuffle=True, drop_last=True)

        self.optimizer = torch.optim.Adam([{'params': self.clf_body.parameters()},
                                           {'params': self.clf_head.parameters()}], lr=self.learn_rate)
        self.CE_criterion = nn.CrossEntropyLoss(reduction='none')

        self.loss_computer = LossComputer(
            criterion=self.CE_criterion,
            is_robust=True,
            dataset={"n_groups": 2, "group_counts": self.group_counts, "group_frac": self.group_frac},
            alpha=self.debias_alpha,
            gamma=self.debias_gamma,
            step_size=0.01,
            normalize_loss=False,
            min_var_weight=0.)


    def fit(self):

        self.clf_init()
        best_obj = -9999
        for epoch in range(self.learn_step):

            self.run_epoch()
            test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val, self.clf_body,
                                                          self.clf_head, self.problem, {'fn_weight': 0., 'fp_weight': 0., 'gamma': 0.})

            # self.logger.info('Epoch = %s,', epoch)
            # self.logger.info("Test result:")
            # self.logger.info("Acc = %s,", test_result['accuracy'])
            # self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", test_result['fpr_diff'], test_result['0_fpr'],
            #                  test_result['1_fpr'])
            # self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", test_result['fnr_diff'], test_result['0_fnr'],
            #                  test_result['1_fnr'])
            # self.logger.info("EO = %s,",
            #                  test_result['1_fnr'] - test_result['0_fnr'] + test_result['0_fpr'] - test_result['1_fpr'])
            # self.logger.info("Obj = %s,", test_result['objective'])
            # self.logger.info("----------------------------")

            cur_EO = abs(test_result['1_fnr'] - test_result['0_fnr']) + abs(test_result['0_fpr'] - test_result['1_fpr'])
            cur_acc = test_result['accuracy']

            # print(cur_acc, cur_EO)

            if test_result['1_fpr'] >= 1. or test_result['0_fpr'] >= 1.:
                continue

            if test_result['1_fnr'] >= 1. or test_result['0_fnr'] >= 1.:
                continue

            cur_obj = self.obj_acc * cur_acc - cur_EO

            # print(cur_obj, cur_acc, cur_EO)
            if cur_obj > best_obj:
                target_network_update(self.clf_body, self.best_clf_body)
                target_network_update(self.clf_head, self.best_clf_head)
                best_obj = cur_obj

            best_test_result = measure_objective_results_body_head(self.x_val, self.y_val, self.z_val,
                                                               self.best_clf_body, self.best_clf_head, None, None)

            self.logger.info('DRO Debias')
            self.logger.info("Step %s Test result:", epoch)
            self.logger.info("Acc = %s,", best_test_result['accuracy'])
            self.logger.info("Fpr = %s, (Fpr0=%s, Fpr1=%s)", best_test_result['fpr_diff'], best_test_result['0_fpr'],
                             best_test_result['1_fpr'])
            self.logger.info("Fnr = %s, (Fnr0=%s, Fnr1=%s)", best_test_result['fnr_diff'], best_test_result['0_fnr'],
                             best_test_result['1_fnr'])
            self.logger.info("EO = %s,",
                             -abs(best_test_result['1_fnr'] - best_test_result['0_fnr']) - abs(
                                 best_test_result['0_fpr'] - best_test_result['1_fpr']))
            self.logger.info("Obj = %s,", best_test_result['objective'])
            self.logger.info("----------------------------")

            save_checkpoint(os.path.join(self.output_dir, "clf_body", "clf_body.pth.tar"),
                            round_index=self.run_num,
                            label_num=0,
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



    def run_epoch(self, log_every=50):

        for batch_idx, (x, y, g) in enumerate(self.train_loader):

            outputs = self.clf_head(self.clf_body(x))
            loss_main = self.loss_computer.loss(outputs, y, g, True)
            self.optimizer.zero_grad()
            loss_main.backward()
            self.optimizer.step()

            if (batch_idx + 1) % log_every == 0:
                # self.loss_computer.log_stats(self.logger, True)
                self.loss_computer.reset_stats()

        if self.loss_computer.batch_count > 0:
            # self.loss_computer.log_stats(self.logger, True)
            self.loss_computer.reset_stats()

        # checkpoint_body_fname = os.path.join(self.output_dir, "clf_body", "clf_body_0.pth.tar")
        # save_checkpoint(checkpoint_body_fname,
        #                 round_index=self.run_num,
        #                 label_num=0,
        #                 clf_body_state_dict=self.clf_body.state_dict(),
        #                 layer_num=self.body_layer_num,
        #                 clf_input_dim=self.x_train.shape[1],
        #                 hidden_dim=self.body_hidden_dim,
        #                 clf_output_dim=self.embedding_dim,
        #                 )
        #
        # checkpoint_head_fname = os.path.join(self.output_dir, "clf_head", "clf_head_0.pth.tar")
        # save_checkpoint(checkpoint_head_fname,
        #                 round_index=self.run_num,
        #                 label_num=0,
        #                 layer_num=self.head_layer_num,
        #                 clf_head_state_dict=self.clf_head.state_dict(),
        #                 clf_input_dim=self.embedding_dim,
        #                 hidden_dim=self.head_hidden_dim,
        #                 clf_output_dim=2,
        #                 )

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
parser = argparse.ArgumentParser("DRO")
parser.add_argument('--hyper', type=str, default='medical_DRO', help='.')
args = parser.parse_args()

def main():

    # load hyperparameters
    json_fname = args.hyper

    import json
    with open(os.path.join("hyperparameters/DRO", json_fname)) as json_file:
        hyperparam = json.load(json_file)

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
