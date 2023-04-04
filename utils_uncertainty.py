import numpy as np
import scipy
import sklearn
import torch

# define function to caluculate the Calibration AUC of a classifier given the predicitons, targets, and confidences
def cal_auc_np(preds, targets, confs):
    # determine whether the predictions are correct
    correct = (preds == targets).astype(float)

    # now use built in methods to calculate the AUROC of a binary classifier
    fpr, tpr, _ = sklearn.metrics.roc_curve(correct, confs)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc



# define function to calculate the Oracle Collaboration Accuracy of a classifier given the predictions, targets, confidences
# and the fraction of examples to defer to the oracle. That is, the accuracy of the classifier when defering some examples
# to the oracle, based on the confidence of the classifier
def oracle_collab_acc_np(preds, targets, confs, oracle_frac, diff=True):
    # determine whether the predictions are correct
    correct = (preds == targets).astype(float)

    # get the number of examples to defer to the oracle
    oracle_num = int(oracle_frac * len(correct))

    # get the examples to defer to the oracle, taking the ones with the lowest confidence
    oracle_ids = np.argsort(confs)[:oracle_num]

    # determine the accuracy of the non-deferred examples
    non_oracle_ids = list(set(range(len(correct))) - set(oracle_ids))
    non_oracle_acc = np.mean(correct[non_oracle_ids])
    if not non_oracle_ids:
        non_oracle_acc = 0.

    # determine the accuracy of the overall system which is the weighted average of the non-deferred examples
    # and the oracle, which is always correct
    oracle_acc = 1.
    overall_acc = (non_oracle_acc * (1. - oracle_frac)) + (oracle_acc * oracle_frac)

    # determine the difference in accuracy between the classifier with and without the oracle
    diff_acc = overall_acc - correct.mean()

    if diff:
        return diff_acc
    else:
        return overall_acc


# define the function to calculate the Oracle Collaboration AUC of a classifier given the predictions, targets, confidences
# and the fraction of examples to defer to the oracle. That is, the AUC of the classifier when defering some examples
# to the oracle, based on the confidence of the classifier
def oracle_collab_auc_np(preds, targets, confs, oracle_frac):
    # print(preds)
    # print(targets)
    # print(confs)
    # determine whether the predictions are correct
    correct = (preds == targets).astype(float)
    # print(correct)

    # get the number of examples to defer to the oracle
    oracle_num = int(oracle_frac * len(correct))
    # print(oracle_num)

    # get the examples to defer to the oracle, taking the ones with the lowest confidence
    oracle_ids = np.argsort(confs.max(axis=-1))[:oracle_num]
    # print(oracle_ids)

    correct[oracle_ids] = 1.

    fpr, tpr, _ = sklearn.metrics.roc_curve(correct, confs.max(axis=-1))

    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

def temper_probs(probs, T):
    # convert probs to logits
    logits = torch.log(probs)
    # temper logits
    logits_tempered = logits / T
    # convert back to probs
    probs_tempered = torch.softmax(logits_tempered, dim=2)
    return probs_tempered

def get_ood_scores(probs):
    entropy = scipy.stats.entropy(probs.numpy(), axis=2)
    msp = probs.numpy().max(axis=2)
    return entropy, msp

def get_ood_detection_roc(ID_score, OOD_score):
    targets = np.concatenate([np.ones(OOD_score.shape[0]), np.zeros(ID_score.shape[0])], axis=0)
    score_vals = np.concatenate([OOD_score, ID_score], axis=0)
    fpr, tpr, _ = sklearn.metrics.roc_curve(targets, score_vals)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc
