from omegaconf import DictConfig, OmegaConf
from .utils import init, Manager, load_logger
from .core import CourseBook
import traceback


def console():
    conf = init()
    logger = load_logger(conf.logger)
    course = CourseBook(conf=conf.data)
    logger.info(f"Task {conf.task} requested using package {conf.package}")
    try:
        if conf.package == "sklearn":
            logger.critical("scikit-learn is not yet supported. dumbass")
        elif conf.package == "torch":
            from .torch import SupervisedLearner, load_learner

            learner = load_learner(conf.learner)

            if conf.task == "fit" or conf.task == "train":
                # load learner, trainer, grader
                from .torch import load_grader, load_trainer

                grader = load_grader(conf.grader, logger)
                trainer = load_trainer(conf.trainer, logger, grader, course, learner)
                trainer.fit()

            if conf.task == "eval" or conf.task == "test":
                # load learner, grader
                from .torch import load_grader

                grader = load_grader(conf.grader, logger, course, learner)
                grader.eval()

            if conf.task == "infer" or conf.task == "predict":
                # load learner, predictor
                pass
        else:
            raise ValueError(f"Not a valid option (package: {conf.package})")
    except:
        logger.exception(traceback.format_exc())
