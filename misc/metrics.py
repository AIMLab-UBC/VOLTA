import numpy as np
import torch
from numpy import copy
from scipy.optimize import linear_sum_assignment as linear_sum_assignment
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, \
    balanced_accuracy_score, f1_score, cluster, v_measure_score, homogeneity_score, completeness_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger is not None:
            self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def remap_label_list(label_list, mapping=None):
    # get unique values
    if mapping is None:
        mapping = {label: i for i, label in enumerate(np.unique(label_list))}
    # copy list
    remaped_labels = copy(label_list)
    # remap labels
    for k, v in mapping.items():
        remaped_labels[label_list == k] = v
    return remaped_labels, mapping


def get_best_cluster_assignment(true_labels, clustering_labels, non_existing_values=None):
    _, _, assignment = _cluster_search_best_assignment(true_labels, clustering_labels,
                                                       non_existing_values=non_existing_values)
    return assignment


def cluter_assignment(true_labels, clustering_labels):
    true_labels, clustering_labels, _ = _cluster_search_best_assignment(true_labels, clustering_labels)
    return true_labels, clustering_labels


# Note: https://datascience.stackexchange.com/a/64208
# Note: true label should start from one
def _cluster_search_best_assignment(true_labels, clustering_labels, non_existing_values=None):
    # remap labels
    true_labels, true_label_to_internal_map = remap_label_list(true_labels)
    clustering_labels, cluster_label_to_internal_map = remap_label_list(clustering_labels)

    max_label = max(np.max(true_labels), np.max(clustering_labels))
    W = np.zeros((max_label + 1, max_label + 1))

    for y_true, y_pred in zip(true_labels, clustering_labels):
        W[y_pred, y_true] += 1

    W = np.max(np.max(W, axis=0), axis=0) - W

    row_ind, col_ind = linear_sum_assignment(W)

    cluster_label_to_true_label_internal_map = {r: c for r, c in zip(row_ind, col_ind)}

    clustering_labels, _ = remap_label_list(clustering_labels, cluster_label_to_true_label_internal_map)

    internal_to_true_label_map = {v: k for k, v in true_label_to_internal_map.items()}
    true_labels, _ = remap_label_list(true_labels, internal_to_true_label_map)
    clustering_labels, _ = remap_label_list(clustering_labels, internal_to_true_label_map)

    cluster_label_final_map = {k: internal_to_true_label_map[cluster_label_to_true_label_internal_map[v]]
                               for k, v in cluster_label_to_internal_map.items()}

    if non_existing_values:
        if not isinstance(non_existing_values, np.ndarray):
            non_existing_values = np.array(non_existing_values)
        non_existing_values = np.unique(non_existing_values)
        non_existing_mapping = dict()
        for internal_k, internal_v in cluster_label_to_true_label_internal_map.items():
            if internal_k not in cluster_label_to_internal_map.values():
                cluster_label = np.random.choice(non_existing_values)
                non_existing_values = non_existing_values[non_existing_values != cluster_label]
                non_existing_mapping[cluster_label] = internal_to_true_label_map[internal_v]
        cluster_label_final_map.update(non_existing_mapping)

    return true_labels, clustering_labels, cluster_label_final_map


def clustering_metrics(true_labels, clustering_labels):
    true_labels, clustering_labels = cluter_assignment(true_labels, clustering_labels)
    acc = accuracy_score(true_labels, clustering_labels)
    bacc = balanced_accuracy_score(true_labels, clustering_labels)
    conf = confusion_matrix(true_labels, clustering_labels)
    rand = adjusted_rand_score(true_labels, clustering_labels)
    ami = adjusted_mutual_info_score(true_labels, clustering_labels)
    fscore = f1_score(true_labels, clustering_labels, average=None)
    purity = purity_score(true_labels, clustering_labels)
    v_measure = v_measure_score(true_labels, clustering_labels)
    homogenity = homogeneity_score(true_labels, clustering_labels)
    completeness = completeness_score(true_labels, clustering_labels)
    return {'accuracy': acc, 'balanced accuracy': bacc, 'adjusted random index': rand,
            'adjusted mutual information': ami, 'fscore': fscore.mean(), 'confidence metrix': conf, 
            'purity': purity, 'v_measure': v_measure, 'homogenity': homogenity, 'completeness': completeness}
