import torch
from csv import DictReader
import os
import json
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import torch

# from utils_pokec import load_pokec as

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def log_init(hyperparam):
    checkpoint_name = hyperparam["checkpoint_name"]
    round_num = hyperparam["round_num"]

    import log_utils, time, glob, logging, sys
    from shutil import copyfile

    fdir0 = os.path.join(hyperparam["log_rootdir"], checkpoint_name)
    log_utils.create_exp_dir(fdir0)

    fdir1 = os.path.join(fdir0, checkpoint_name + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    log_utils.create_exp_dir(fdir1, scripts_to_save=glob.glob('*.py'))

    for round_index in range(round_num):
        fdir2 = os.path.join(fdir1, "round" + str(round_index))
        log_utils.create_exp_dir(fdir2)
        fdir3 = os.path.join(fdir2, "clf")
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


class FairnessProblem:
    def __init__(self,
                 x,
                 y,
                 protected_index,
                 test_size=0.3,
                 val_size=0.5,
                 ):
        self.protected_index = protected_index
        self.test_size = test_size
        self.val_size = val_size
        self.X = np.array(x) # torch.tensor(x)
        self.Y = np.array(y) # torch.tensor(y)


def process_line(filters, headers_integer_values, headers_operation, line):
    for f in filters:
        if f.get('values', None):
            if line[f['header']] in f['values']:
                return
        if f.get('range', None):
            lt = float(f['range'].get('lt', -float('inf')))
            gt = float(f['range'].get('gt', float('inf')))
            if f['range']['type'].lower() == 'and':
                if float(line[f['header']]) < lt and float(line[f['header']]) > gt:
                    return
            else:
                if float(line[f['header']]) < lt or float(line[f['header']]) > gt:
                    return

    line_copy = line.copy()
    for h in headers_integer_values:
        if line[h['header']] not in h.keys():
            return
        line_copy[h['header']] = h[line[h['header']]]

    for o in headers_operation:
        if o.get('divide', None):
            line_copy[o['header']] = float(line[o['header']])/float(o['divide'])

    return line_copy


def load_problem_from_options(options):
    # file = open(options_file).read()
    # print('Setting up the problem according to this option file:')
    # print(file)
    # options = json.loads(file)
    # print(options)
    data_file = open(os.path.dirname(__file__) + '/datasets/' + options['data_set'] + '/' + options['file'])
    headers = options['data_headers'].split(',')
    protected = options['protected']
    tag = options['tag']
    test_size = float(options['test_size'])
    val_size = float(options['val_size'])
    filters = options['filters']

    headers_integer_values = options['headers_integer_values']
    headers_operation = options['headers_operation']
    file_reader = DictReader(data_file)

    protected_index = headers.index(protected)
    x = []
    y = []
    number_of_lines = 0
    positive_data = 0
    for line in file_reader:
        number_of_lines += 1
        processed_line = process_line(filters, headers_integer_values, headers_operation, line)
        if processed_line:
            line_data = [float(processed_line[h]) for h in headers]
            line_data.append(1.)
            x.append(line_data)
            line_tag = int(processed_line[tag])
            y.append(line_tag)
            if line_tag == 1:
                positive_data += 1

    print('\ndataset size: ' + str(number_of_lines) + '\ndataset used: ' + str(len(x)))
    print('positive percentage: ' + str(positive_data*1./len(x)) + '\nnegative percentage: ' + str((len(x) - positive_data)*1./len(x)))

    group0_num = (np.array(x)[:, protected_index] == 0).sum()
    print('Group 0 percentage: ' + str(group0_num*1./len(x)) + '\nGroup 1 percentage: ' + str((len(x) - group0_num)*1./len(x)))

    return FairnessProblem(
        x=x,
        y=y,
        protected_index=protected_index,
        test_size=test_size,
        val_size=val_size,
    )


def load_data(problem, run_num=0):

    # print(problem.X.shape)
    # print(problem.X[0])
    # return

    Z = problem.X[:, problem.protected_index].astype(np.int)
    X = np.concatenate((problem.X[:, :problem.protected_index], problem.X[:, problem.protected_index+1:]), axis=1)
    Y = problem.Y

    # minority_index = np.where(Z==0)[0]
    # minority_index_remove = minority_index[0:minority_index.shape[0]]
    # index_keep = list(set(range(0, X.shape[0])) - set(minority_index_remove))

    # majority_index = np.where(Y==1)[0]
    # majority_index_remove = majority_index[0:majority_index.shape[0]//4*3]
    # index_keep = list(set(range(0, X.shape[0])) - set(majority_index_remove))
    #
    # X = X[index_keep]
    # Y = Y[index_keep]
    # Z = Z[index_keep]

    X = StandardScaler().fit_transform(X)

    # print("Majority num:", Z[Z==1].shape[0])
    # print("Minority num:", Z[Z==0].shape[0])

    # print("Positive num:", Y[Y==1].shape[0])
    # print("Negative num:", Y[Y==0].shape[0])

    x_train_all, x_test, y_train_all, y_test, z_train_all, z_test = train_test_split(X, Y, Z,
                                                                                     test_size=problem.test_size,
                                                                                     random_state=run_num)

    x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train_all, y_train_all, z_train_all,
                                                                      test_size=problem.val_size,
                                                                      random_state=0) # , random_state=fold)

    y_train = y_train.astype(np.int)
    y_val = y_val.astype(np.int)
    y_test = y_test.astype(np.int)

    # print(x_train.shape, y_train.shape, z_train.shape)

    # print(train_loader)


    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.int)
    z_train = torch.from_numpy(z_train).type(torch.int)

    return x_train, x_val, x_test, y_train, y_val, y_test, z_train, z_val, z_test


#
# from fairness_data import measures


def get_positive_samples(x_buf, y):
    mask = np.where(y == 1)[0]
    x_pos = [x[mask] for x in x_buf]
    return x_pos, mask


def get_negative_samples(x_buf, y):
    mask = np.where(y == 0)[0]
    x_neg = [x[mask] for x in x_buf]
    return x_neg, mask


def get_positive_examples(x, y):
    mask = np.where(y == 1)[0]
    return x[mask], mask


def get_negative_examples(x, y):
    mask = np.where(y == 0)[0]
    return x[mask], mask


def split_by_protected_value(x, y, z):
    x_1 = x[np.where(z == 1)]
    y_1 = y[np.where(z == 1)]
    x_0 = x[np.where(z == 0)]
    y_0 = y[np.where(z == 0)]
    return x_1, y_1, x_0, y_0

def measure_objective_results_nobody(x_test, y_test, z_test, w, b, problem, hyper):

    x_torch = torch.from_numpy(x_test).type(torch.float)
    y_hat = torch.softmax(torch.matmul(x_torch, w) + b, axis=1).argmax(dim=1).detach().numpy()

    y_1, y_1_hat, y_0, y_0_hat = split_by_protected_value(y_test, y_hat, z_test)
    fnr1 = (y_1_hat[y_1 == 1] == 0).sum() * 1. / (y_1 == 1).sum()
    fnr0 = (y_0_hat[y_0 == 1] == 0).sum() * 1. / (y_0 == 1).sum()
    fpr1 = (y_1_hat[y_1 == 0] == 1).sum() * 1. / (y_1 == 0).sum()
    fpr0 = (y_0_hat[y_0 == 0] == 1).sum() * 1. / (y_0 == 0).sum()
    acc = (y_hat == y_test).sum()*1./y_test.shape[0]
    fpr_diff = abs(fpr1 - fpr0)
    fnr_diff = abs(fnr1 - fnr0)

    # all_measures = measures(y_test, y_hat)
    # _1_measures = measures(y_1, y_1_hat)
    # _0_measures = measures(y_0, y_0_hat)
    # fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
    # fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
    return {
        'accuracy': acc,
        '1_fpr':    fpr1,
        '1_fnr':    fnr1,
        '0_fpr':    fpr0,
        '0_fnr':    fnr0,
        'fpr_diff': fpr_diff,
        'fnr_diff': fnr_diff,
        'objective': acc + (1-fpr_diff) + (1-fnr_diff),
        # 'fn_weight': hyper['fn_weight'],
        # 'fp_weight': hyper['fp_weight'],
        # 'gamma': hyper['gamma'],
    }

def measure_objective_results_torch(x_test, y_test, z_test, classifier_body, w, b, problem, hyper):

    x_test_head = classifier_body(torch.from_numpy(x_test).type(torch.float))
    y_hat = torch.softmax(torch.matmul(x_test_head, w) + b, axis=1).argmax(dim=1).detach().numpy()
    y_1, y_1_hat, y_0, y_0_hat = split_by_protected_value(y_test, y_hat, z_test)
    fnr1 = (y_1_hat[y_1 == 1] == 0).sum() * 1. / (y_1 == 1).sum()
    fnr0 = (y_0_hat[y_0 == 1] == 0).sum() * 1. / (y_0 == 1).sum()
    fpr1 = (y_1_hat[y_1 == 0] == 1).sum() * 1. / (y_1 == 0).sum()
    fpr0 = (y_0_hat[y_0 == 0] == 1).sum() * 1. / (y_0 == 0).sum()
    acc = (y_hat == y_test).sum()*1./y_test.shape[0]
    fpr_diff = abs(fpr1 - fpr0)
    fnr_diff = abs(fnr1 - fnr0)

    # all_measures = measures(y_test, y_hat)
    # _1_measures = measures(y_1, y_1_hat)
    # _0_measures = measures(y_0, y_0_hat)
    # fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
    # fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
    return {
        'accuracy': acc,
        '1_fpr':    fpr1,
        '1_fnr':    fnr1,
        '0_fpr':    fpr0,
        '0_fnr':    fnr0,
        'fpr_diff': fpr_diff,
        'fnr_diff': fnr_diff,
        'objective': acc + (1-fpr_diff) + (1-fnr_diff),
        # 'fn_weight': hyper['fn_weight'],
        # 'fp_weight': hyper['fp_weight'],
        # 'gamma': hyper['gamma'],
    }


def estimate_objective(x_test, y_test, z_prob_test, classifier_body, w, b, problem, hyper):

    with torch.no_grad():
        z_prob0 = z_prob_test[:, 0]
        z_prob1 = z_prob_test[:, 1]

        x_head = classifier_body(torch.from_numpy(x_test).type(torch.float))
        y_prob1 = torch.softmax(torch.matmul(x_head, w) + b, axis=1)[:, 1]
        y_prob0 = 1 - y_prob1
        y_hat = torch.softmax(torch.matmul(x_head, w) + b, axis=1).argmax(dim=1).detach().numpy()

        pos_buf, pos_index = get_positive_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        neg_buf, neg_index = get_negative_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        x_pos, y_prob0_pos, y_prob1_pos, z_prob0_pos, z_prob1_pos = pos_buf
        x_neg, y_prob0_neg, y_prob1_neg, z_prob0_neg, z_prob1_neg = neg_buf

        acc = (y_hat == y_test).sum() * 1. / y_test.shape[0]
        fpr1 = (y_prob1_neg.T * z_prob1_neg).T.sum(axis=0) / z_prob1_neg.sum()
        fpr0 = (y_prob1_neg.T * z_prob0_neg).T.sum(axis=0) / z_prob0_neg.sum()
        fnr1 = (y_prob0_pos.T * z_prob1_pos).T.sum(axis=0) / z_prob1_pos.sum()
        fnr0 = (y_prob0_pos.T * z_prob0_pos).T.sum(axis=0) / z_prob0_pos.sum()
        fpr_diff = abs(fpr1 - fpr0)
        fnr_diff = abs(fnr1 - fnr0)

        return {
            'accuracy': acc,
            '1_fpr': fpr1,
            '1_fnr': fnr1,
            '0_fpr': fpr0,
            '0_fnr': fnr0,
            'fpr_diff': fpr_diff,
            'fnr_diff': fnr_diff,
            'objective': acc * hyper["obj_acc"] + hyper["obj_fpr"] * (1 - fpr_diff) + hyper["obj_fnr"] * (1 - fnr_diff),
            # 'objective': (1 - acc) + fpr_diff + fnr_diff,
            # 'fn_weight': hyper['fn_weight'],
            # 'fp_weight': hyper['fp_weight'],
            # 'gamma': hyper['gamma'],
        }



def save_checkpoint(fname, **kwargs):

    checkpoint = {}

    for key, value in kwargs.items():
        checkpoint[key] = value
        # setattr(self, key, value)

    torch.save(checkpoint, fname)

def load_checkpoint(fname):
    return torch.load(fname)


def measure_objective_results_body_head(x_test, y_test, z_test, clf_body, clf_head, problem, hyper):

    with torch.no_grad():
        if clf_body is None:
            x_test_head = torch.from_numpy(x_test).type(torch.float)
        else:
            with torch.no_grad():
                x_test_head = clf_body(torch.from_numpy(x_test).type(torch.float))

        y_prob = clf_head(x_test_head)
        if torch.is_tensor(y_prob):
            y_hat = y_prob.argmax(dim=1).detach().numpy()
        else:
            y_hat = y_prob.argmax(axis=1)

        # print(y_hat.sum())

        y_1, y_1_hat, y_0, y_0_hat = split_by_protected_value(y_test, y_hat, z_test)
        fnr1 = (y_1_hat[y_1 == 1] == 0).sum() * 1. / (y_1 == 1).sum()
        fnr0 = (y_0_hat[y_0 == 1] == 0).sum() * 1. / (y_0 == 1).sum()
        fpr1 = (y_1_hat[y_1 == 0] == 1).sum() * 1. / (y_1 == 0).sum()
        fpr0 = (y_0_hat[y_0 == 0] == 1).sum() * 1. / (y_0 == 0).sum()

        # print(fnr1, fnr0, fpr1, fpr0)

        acc = (y_hat == y_test).sum()*1./y_test.shape[0]
        fpr_diff = abs(fpr1 - fpr0)
        fnr_diff = abs(fnr1 - fnr0)

        # all_measures = measures(y_test, y_hat)
        # _1_measures = measures(y_1, y_1_hat)
        # _0_measures = measures(y_0, y_0_hat)
        # fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
        # fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
        return {
            'accuracy': acc,
            '1_fpr':    fpr1,
            '1_fnr':    fnr1,
            '0_fpr':    fpr0,
            '0_fnr':    fnr0,
            'fpr_diff': fpr_diff,
            'fnr_diff': fnr_diff,
            'objective': acc + (1-fpr_diff) + (1-fnr_diff),
            'equal_odds': -abs(fnr1 - fnr0) - abs(fpr0 - fpr1),
            # 'equal_odds': fnr1 - fnr0 + fpr0 - fpr1,
            # 'fn_weight': hyper['fn_weight'],
            # 'fp_weight': hyper['fp_weight'],
            # 'gamma': hyper['gamma'],
        }

def estimate_objective_body_head(x_test, y_test, z_prob_test, clf_body, clf_head, problem, hyper):

    with torch.no_grad():
        z_prob0 = z_prob_test[:, 0]
        z_prob1 = z_prob_test[:, 1]

        if clf_body is None:
            x_head = torch.from_numpy(x_test).type(torch.float)
        else:
            with torch.no_grad():
                x_head = clf_body(torch.from_numpy(x_test).type(torch.float))

        y_prob1 = torch.softmax(clf_head(x_head), dim=1)[:, 1]
        y_prob0 = 1 - y_prob1
        y_hat = clf_head(x_head).argmax(dim=1).detach().numpy()

        pos_buf, pos_index = get_positive_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        neg_buf, neg_index = get_negative_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        x_pos, y_prob0_pos, y_prob1_pos, z_prob0_pos, z_prob1_pos = pos_buf
        x_neg, y_prob0_neg, y_prob1_neg, z_prob0_neg, z_prob1_neg = neg_buf

        acc = (y_hat == y_test).sum() * 1. / y_test.shape[0]
        fpr1 = (y_prob1_neg.T * z_prob1_neg).T.sum(axis=0) / z_prob1_neg.sum()
        fpr0 = (y_prob1_neg.T * z_prob0_neg).T.sum(axis=0) / z_prob0_neg.sum()
        fnr1 = (y_prob0_pos.T * z_prob1_pos).T.sum(axis=0) / z_prob1_pos.sum()
        fnr0 = (y_prob0_pos.T * z_prob0_pos).T.sum(axis=0) / z_prob0_pos.sum()
        fpr_diff = abs(fpr1 - fpr0)
        fnr_diff = abs(fnr1 - fnr0)

        return {
            'accuracy': acc,
            '1_fpr': fpr1,
            '1_fnr': fnr1,
            '0_fpr': fpr0,
            '0_fnr': fnr0,
            'fpr_diff': fpr_diff,
            'fnr_diff': fnr_diff,
            'objective': acc * hyper["obj_acc"] + hyper["obj_fpr"] * (1 - fpr_diff) + hyper["obj_fnr"] * (1 - fnr_diff),
            # 'objective': (1 - acc) + fpr_diff + fnr_diff,
            # 'fn_weight': hyper['fn_weight'],
            # 'fp_weight': hyper['fp_weight'],
            # 'gamma': hyper['gamma'],
        }


def measure_objective_results_body_head_eop(x_test, y_test, z_test, clf_body, clf_head, problem, hyper):

    with torch.no_grad():
        if clf_body is None:
            x_test_head = torch.from_numpy(x_test).type(torch.float)
        else:
            x_test_head = clf_body(torch.from_numpy(x_test).type(torch.float))

        y_hat = clf_head(x_test_head).argmax(dim=1).detach().numpy()
        y_1, y_1_hat, y_0, y_0_hat = split_by_protected_value(y_test, y_hat, z_test)
        fnr1 = (y_1_hat[y_1 == 1] == 0).sum() * 1. / (y_1 == 1).sum()
        fnr0 = (y_0_hat[y_0 == 1] == 0).sum() * 1. / (y_0 == 1).sum()
        fpr1 = (y_1_hat[y_1 == 0] == 1).sum() * 1. / (y_1 == 0).sum()
        fpr0 = (y_0_hat[y_0 == 0] == 1).sum() * 1. / (y_0 == 0).sum()

        acc = (y_hat == y_test).sum()*1./y_test.shape[0]
        fpr_diff = abs(fpr1 - fpr0)
        fnr_diff = abs(fnr1 - fnr0)
        # eop = min(1-fnr0, 1-fnr1) / max(1-fnr0, 1-fnr1)

        # all_measures = measures(y_test, y_hat)
        # _1_measures = measures(y_1, y_1_hat)
        # _0_measures = measures(y_0, y_0_hat)
        # fpr_diff = abs(_1_measures['fpr'] - _0_measures['fpr'])
        # fnr_diff = abs(_1_measures['fnr'] - _0_measures['fnr'])
        return {
            'accuracy': acc,
            '1_fpr':    fpr1,
            '1_fnr':    fnr1,
            '0_fpr':    fpr0,
            '0_fnr':    fnr0,
            'fpr_diff': fpr_diff,
            'fnr_diff': fnr_diff,
            'objective': acc + fnr_diff,
            # 'eop': eop,
            # 'equal_odds': fnr1 - fnr0 + fpr0 - fpr1,
            # 'fn_weight': hyper['fn_weight'],
            # 'fp_weight': hyper['fp_weight'],
            # 'gamma': hyper['gamma'],
        }

def estimate_objective_body_head_eop(x_test, y_test, z_prob_test, clf_body, clf_head, problem, hyper):

    with torch.no_grad():
        z_prob0 = z_prob_test[:, 0]
        z_prob1 = z_prob_test[:, 1]

        if clf_body is None:
            x_head = torch.from_numpy(x_test).type(torch.float)
        else:
            x_head = clf_body(torch.from_numpy(x_test).type(torch.float))

        y_prob1 = torch.softmax(clf_head(x_head), dim=1)[:, 1]
        y_prob0 = 1 - y_prob1
        y_hat = clf_head(x_head).argmax(dim=1).detach().numpy()

        pos_buf, pos_index = get_positive_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        neg_buf, neg_index = get_negative_samples([x_test, y_prob0, y_prob1, z_prob0, z_prob1], y_test)
        x_pos, y_prob0_pos, y_prob1_pos, z_prob0_pos, z_prob1_pos = pos_buf
        x_neg, y_prob0_neg, y_prob1_neg, z_prob0_neg, z_prob1_neg = neg_buf

        acc = (y_hat == y_test).sum() * 1. / y_test.shape[0]
        fpr1 = (y_prob1_neg.T * z_prob1_neg).T.sum(axis=0) / z_prob1_neg.sum()
        fpr0 = (y_prob1_neg.T * z_prob0_neg).T.sum(axis=0) / z_prob0_neg.sum()
        fnr1 = (y_prob0_pos.T * z_prob1_pos).T.sum(axis=0) / z_prob1_pos.sum()
        fnr0 = (y_prob0_pos.T * z_prob0_pos).T.sum(axis=0) / z_prob0_pos.sum()
        fpr_diff = abs(fpr1 - fpr0)
        fnr_diff = abs(fnr1 - fnr0)
        # eop = min(fnr0, fnr1) / max(fnr0, fnr1)

        return {
            'accuracy': acc,
            '1_fpr': fpr1,
            '1_fnr': fnr1,
            '0_fpr': fpr0,
            '0_fnr': fnr0,
            'fpr_diff': fpr_diff,
            'fnr_diff': fnr_diff,
            'objective': acc * hyper["obj_acc"] + fnr_diff,
            # 'objective': (1 - acc) + fpr_diff + fnr_diff,
            # 'fn_weight': hyper['fn_weight'],
            # 'fp_weight': hyper['fp_weight'],
            # 'gamma': hyper['gamma'],
        }

# save_checkpoint('./a.pth.tar', a=1)
# load_problem_from_options('./options/adult.json')
# torch.Variable()
