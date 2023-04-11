from typing import Sequence, Union
from numpy import ndarray
import sklearn.metrics as mt

metric = {
    "acc" : mt.accuracy_score,
    "f1" : mt.f1_score,
    "top_k_acc" : mt.top_k_accuracy_score,
    "fbeta" : mt.fbeta_score,
    "hamming" : mt.hamming_loss,
    "hinge" : mt.hinge_loss,
    "jaccard" : mt.jaccard_score,
    "log" : mt.log_loss,
    "precision" : mt.precision_score,
    "recall" : mt.recall_score,
    "roc_auc" : mt.roc_auc_score,
    "pr_curve" : mt.precision_recall_curve,
    "roc_curve" : mt.roc_curve,
    "01" : mt.zero_one_loss,
    "max" : mt.max_error,
    "mae" : mt.mean_absolute_error,
    "mse" : mt.mean_squared_error,
    "msle" : mt.mean_squared_log_error,
    "mdae" : mt.median_absolute_error,
}

class Grader():

    @staticmethod
    def eval(preds, gt, methods:Union[str, Sequence[str]], **kwargs):
        
        if isinstance(method, str):
            method = [method]

        if not isinstance(preds, ndarray):
            preds = preds.numpy()

        if not isinstance(gt, ndarray):
            gt = gt.numpy()

        ret = []

        for method in methods:
            ret.append(metric[method](gt, preds, **kwargs))

        return ret
