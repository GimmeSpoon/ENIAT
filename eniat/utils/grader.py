from typing import Sequence, Union, TypeVar, Callable
from numpy import ndarray
import sklearn.metrics as mt
from omegaconf import DictConfig

T_co = TypeVar('T_co', covariant=True)

sk_metric = {
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

def sk_eval(preds, gt, methods:Union[str, Sequence[str]], **kwargs):
        
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

class Grader():
    
    class Result():
        def __init__(self) -> None:
            pass
    
    def __init__(self, conf:DictConfig, methods:Union[str, Callable, Sequence[Union[str, Callable]]]) -> None:
        self.unit = conf.unit
        self.interval = conf.interval
        self.methods = []
        self.append_metric(methods)

    def append_metric(self, method:Union[str, Callable, Sequence[Union[str, Callable]]]):
        if isinstance(methods, Callable):
            methods = [methods]
        elif isinstance(methods, str):
            methods = [sk_metric[methods]]
        else:
            methods = []
            for method in methods:
                if isinstance(methods, Callable):
                    methods.append(method)
                else:
                    methods.append(sk_metric[methods])

    def compute(self, prediction:T_co, ground_truth:T_co, **kwargs) -> dict:
        result = {}
        for method in self.methods:
            result[method.__name__]=method(prediction, ground_truth, **kwargs)
        return result

    def __call__(self, prediction:T_co, ground_truth:T_co, **kwargs):
        self.compute(prediction, ground_truth)