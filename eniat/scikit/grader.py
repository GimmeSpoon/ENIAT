from typing import Sequence, Union, TypeVar, Callable, Literal
from numpy import ndarray
import sklearn.metrics as mt
from ..core import Learner, Course
from omegaconf import DictConfig

T_co = TypeVar("T_co", covariant=True)
L = TypeVar("L", bound=Learner)
C = TypeVar("C", bound=Course)

sk_metric = {
    "acc": mt.accuracy_score,
    "f1": mt.f1_score,
    "top_k_acc": mt.top_k_accuracy_score,
    "fbeta": mt.fbeta_score,
    "hamming": mt.hamming_loss,
    "hinge": mt.hinge_loss,
    "jaccard": mt.jaccard_score,
    "log": mt.log_loss,
    "precision": mt.precision_score,
    "recall": mt.recall_score,
    "roc_auc": mt.roc_auc_score,
    "pr_curve": mt.precision_recall_curve,
    "roc_curve": mt.roc_curve,
    "01": mt.zero_one_loss,
    "max": mt.max_error,
    "mae": mt.mean_absolute_error,
    "mse": mt.mean_squared_error,
    "msle": mt.mean_squared_log_error,
    "meae": mt.median_absolute_error,
}


def sk_eval(preds, gt, methods: Union[str, Sequence[str]], **kwargs):

    if isinstance(method, str):
        methods = [method]

    if not isinstance(preds, ndarray):
        preds = preds.numpy()

    if not isinstance(gt, ndarray):
        gt = gt.numpy()

    ret = []

    for method in methods:
        ret.append(sk_metric[method](gt, preds, **kwargs))

    return ret
