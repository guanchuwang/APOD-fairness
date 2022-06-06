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
from utils import load_data, save_checkpoint, load_checkpoint, save_checkpoint


import random
from sklearn.metrics.pairwise import euclidean_distances
import os, shutil

from main_apd import mlp, Agent as APDAgent


class Agent(APDAgent):

    def __init__(self, problem, hyperparam, logger):
        super().__init__(problem, hyperparam, logger)

    def load_clf_body(self):

        clfb_checkpoint = load_checkpoint(os.path.join(self.output_dir, "clf_body", "clf_body_0.pth.tar"))

        self.clf_body = mlp(input_dim=clfb_checkpoint["clf_input_dim"], output_dim=clfb_checkpoint["clf_output_dim"],
                            layer_num=clfb_checkpoint["layer_num"], hidden_dim=clfb_checkpoint["hidden_dim"])
        self.clf_body.load_state_dict(clfb_checkpoint["clf_body_state_dict"])

    def load_clf_head(self, label_num):

        cp_fullname = os.path.join(self.output_dir, "clf_head", "clf_head_" + str(label_num) + ".pth.tar")
        checkpoint = load_checkpoint(cp_fullname)

        self.clf_head = mlp(input_dim=checkpoint["clf_input_dim"], output_dim=checkpoint["clf_output_dim"],
                            layer_num=checkpoint["layer_num"], hidden_dim=checkpoint["hidden_dim"])

        self.clf_head.load_state_dict(checkpoint["clf_head_state_dict"])

    def generate_sensitive_label(self, label_number):
        index_all = [index for index in range(self.x_train.shape[0])]
        label_index = np.random.choice(index_all, label_number)
        self.if_label[label_index] = True
        print(self.if_label.sum())

    def fit(self):

        self.clf_body_init()
        self.clf_head_init()
        self.load_clf_body()
        self.load_clf_head(label_num=0)
        # self.clf_body_fit(MAX_step=self.body_fit_step)

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

        checkpoint_fname = os.path.join(self.output_dir, "clf_head", "clf_head_0.pth.tar")
        save_checkpoint(checkpoint_fname,
                        round_index=self.run_num,
                        label_num=self.label_budget,
                        layer_num=self.head_layer_num,
                        clf_head_state_dict=self.clf_head.state_dict(),
                        clf_input_dim=self.embedding_dim,
                        hidden_dim=self.head_hidden_dim,
                        clf_output_dim=2,
                        )

        self.clf_head_init()
        # self.warm_up()
        self.generate_sensitive_label(label_number=self.label_budget)
        self.clf_head_init()
        test_result = self.clf_head_fit(MAX_step=self.learn_step, weight=self.debias_weight, gamma=self.debias_gamma)
        # test_measurment_buf.append(test_result)

        checkpoint_fname = os.path.join(self.output_dir, "clf_head", "clf_head_" + str(self.label_budget) + ".pth.tar")
        save_checkpoint(checkpoint_fname,
                        round_index=self.run_num,
                        label_num=self.label_budget,
                        layer_num=self.head_layer_num,
                        clf_head_state_dict=self.clf_head.state_dict(),
                        clf_input_dim=self.embedding_dim,
                        hidden_dim=self.head_hidden_dim,
                        clf_output_dim=2,
                        )

        # self.logger.info('Label Number = %s,', label_num)
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

    fdir1 = os.path.join(hyperparam["log_rootdir"], "rs_fairness")
    log_utils.create_exp_dir(fdir1)

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



# bank
# hyperparam_fname = "log/checkpoint_bank/checkpoint_bank-20210626-211903/bank.json" # 30 lambda 0.001
# hyperparam_fname = "log/checkpoint_bank/checkpoint_bank-20210626-211400/bank.json" # 30 lambda 0.01
# hyperparam_fname = "log/checkpoint_bank/checkpoint_bank-20210624-151542/bank.json" # 30 lambda 0.1
# hyperparam_fname = "log/checkpoint_bank/checkpoint_bank-20210624-151659/bank.json" # 30 lambda 0.5
# hyperparam_fname = "log/checkpoint_bank/checkpoint_bank-20210624-151336/bank.json" # 30 lambda 1.

# law_race
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-111835/ucla_law_race_preprocess.json" # lambda 0.00001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-210855/ucla_law_race_preprocess.json" # lambda 0.0001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-111653/ucla_law_race_preprocess.json" # lambda 0.0002
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-111558/ucla_law_race_preprocess.json" # lambda 0.0003
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-110815/ucla_law_race_preprocess.json" # lambda 0.0004
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162540/ucla_law_race_preprocess.json" # lambda 0.0005
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-161439/ucla_law_race_preprocess.json" # lambda 0.001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-162952/ucla_law_race_preprocess.json" # lambda 0.005

# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210624-160932/ucla_law_race_preprocess.json" # hidden_dim 16 lambda 0.001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-102911/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 32 lambda 0.001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-104218/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 64 lambda 0.0001
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-104829/ucla_law_race_preprocess.json" # embedding_dim 64 hidden_dim 64 lambda 0.0005
# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-105945/ucla_law_race_preprocess.json" # embedding_dim 16 hidden_dim 16 lambda 0.0001

# hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-112036/ucla_law_race_preprocess.json" # seed 1 embedding_dim 32 hidden_dim 32 lambda 0.0001
# hyperparam_fname = "log/checkpoint_ucla_law_gender/checkpoint_ucla_law_gender-20210627-122413/ucla_law_sex_preprocess.json" # seed 1 embedding_dim 32 hidden_dim 32 lambda 0.0001
hyperparam_fname = "log/checkpoint_ucla_law_race/checkpoint_ucla_law_race-20210627-205407/ucla_law_race_preprocess.json" # seed 1 embedding_dim 32 hidden_dim 32 lambda 0.001


### Medical
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210624-114757/medical.json" # apd lambda 0.001
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210624-115807/medical.json" # apd lambda 0.005
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210624-114335/medical.json" # best apd lambda 0.01
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210626-215613/medical.json" # best apd lambda 0.02
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210626-215334/medical.json" # best apd lambda 0.05
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210624-115243/medical.json" # apd lambda 0.1
# hyperparam_fname = "log/checkpoint_medical/checkpoint_medical-20210626-214032/medical.json" # apd lambda 0.2


# adult
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210624-164148/adult_preprocess.json" # lambda 0.01
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210624-165505/adult_preprocess.json" # lambda 0.05
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210624-164810/adult_preprocess.json" # lambda 0.1

# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-192429/adult_preprocess.json" # lambda 0.01
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-211903/adult_preprocess.json" # lambda 0.02
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-212415/adult_preprocess.json" # lambda 0.03
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-191308/adult_preprocess.json" # lambda 0.05
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-194213/adult_preprocess.json" # lambda 0.1
# hyperparam_fname = "log/checkpoint_adult/checkpoint_adult-20210625-210729/adult_preprocess.json" # lambda 1


def main():

    import json
    with open(hyperparam_fname) as json_file:
        hyperparam = json.load(json_file)

    json_fname = hyperparam_fname.split('/')[-1]
    hyperparam["log_rootdir"] = hyperparam_fname.replace(json_fname, "")

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
