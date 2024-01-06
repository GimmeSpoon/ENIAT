from typing import Literal, Union, Callable, TypeVar, Sequence
import numpy as np
import pickle as pk
from omegaconf import DictConfig
from ..utils.conf import instantiate, load_class

T_co = TypeVar("T_co", covariant=True)


class ConfigError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def get_course_instance(cfg: DictConfig, log=None, as_list: bool = False):
    if as_list:
        _courses = []
    else:
        _courses = CourseBook()

    for label in cfg:
        if "cls" in cfg[label] and cfg[label]["cls"]:
            _courses.append(
                Course(
                    label,
                    instantiate(cfg[label]),
                    transform=instantiate(cfg[label]["transform"], _partial=True) if "transform" in cfg[label] else None,
                    cache=cfg[label]["cache"] if "cache" in cfg[label] else False,
                )
            )
            if log:
                log.info(f"'{label}' data is loaded.")
        elif "path" in cfg[label] and cfg[label]["path"]:
            _courses.append(
                Course(
                    label,
                    data=batch_load(cfg[label]["path"], cfg[label].type),
                    transform=instantiate(cfg[label]["transform"], _partial=True) if "transform" in cfg[label] else None,
                    cache=cfg[label["cache"]] if "cache" in cfg[label] else False,
                )
            )
            if log:
                log.info(f"'{label}' data is loaded.")
        else:
            if log:
                log.warning(
                    f"Data(:{label}) is not loaded because the path of data is not specified."
                )
            return
    if not len(_courses):
        log.error("No data is given! Terminating the job...")
        return
    return _courses


def batch_load(paths: Union[str, Sequence[str]], type: Literal["csv", "npy", "pkl"]):
    def np_load(path: str, type: Literal["csv", "npy", "pkl"]):
        if type == "csv":
            return np.loadtxt(path, delimiter=",")
        if type == "npy":
            return np.load(path)
        if type == "pkl":
            return np.load(path, allow_pickle=True)

    if isinstance(paths, str):
        return np_load(paths, type)
    else:
        ret = None
        for path in paths:
            stack = np_load(path, type)
            if ret is None:
                ret = np.empty((0, *stack.shape[1:]))
            ret = np.concatenate((ret, stack))
        return ret


class Course:
    def __init__(
        self,
        label: str,
        data: T_co,
        target: T_co = None,
        transform: Callable = None,
        cache: bool = False,
    ) -> None:
        self.label = label
        self.data = data
        self.target = target
        self.transform = transform
        self.cache = cache

        if target and (t := len(target)) != (d := len(data)):
            raise ValueError(f"The data has different size({d}) with the target({t}).")

    def has_target(self) -> bool:
        return bool(self.target)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> T_co:
        if self.transform:
            x = self.transform(self.data[idx])
        else:
            x = self.data[idx]

        if self.target:
            return x, self.target[idx]
        else:
            return x

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Course):
            return self.label == __value.label
        else:
            raise TypeError(
                "class <Course> does not support comaprison between other types than 'Course'."
            )

    def __iter__(self):
        return iter(zip(self.data, self.target)) if self.target else iter(self.data)

    def __repr__(self) -> str:
        return (
            f"{self.label} ({self.__len__()}/{len(self.target) if self.target else 0})"
        )


class CourseBook:
    def __init__(
        self,
        courses: Union[Course, Sequence[Course]] = None,
        transforms: Union[Callable, Sequence[Callable], dict] = None,
        preprocess: bool = False,
        conf: DictConfig = None,
    ) -> None:

        self.conf = conf

        if courses:
            if isinstance(courses, Course):
                self.__courses = {courses.label: courses}
            else:
                self.__courses = {course.label: course for course in courses}
        else:
            self.__courses = {}

        if transforms:
            if callable(transforms):
                for cr in self.__courses.values():
                    if preprocess:
                        cr.data = transforms(cr.data)
                    else:
                        cr.transform = transforms
            elif isinstance(transforms, dict):
                for key, tr in transforms.items():
                    if preprocess:
                        self.__courses[key].data = tr(self.__courses[key])
                    else:
                        self.__courses[key].transform = tr
            else:  # seq(callable)
                if len(transforms) == len(courses):
                    for i, cr in enumerate(self.__courses.values()):
                        if preprocess:
                            cr.data = transforms(cr.data)
                        else:
                            cr.transform = transforms[i]
                else:
                    raise ValueError(
                        "Sequences of Courses and Transforms must have the same length"
                    )

    def load(self) -> None:
        course_list = get_course_instance(self.conf, as_list=True)
        for course in course_list:
            self.__courses[course.label] = course

    def get(self, label: str):
        return self.__courses[label]

    def append(self, course: Course = None, label: str = None, data=None):
        if course:
            self.__courses[course.label] = course
        elif data and label:
            self.__courses[label] = data
        else:
            raise ValueError(
                "At least one of 'course' or a pair of 'label' and 'data' must be given"
            )
        return self

    def __getitem__(self, key) -> Course:
        return self.__courses[key]

    def __len__(self) -> int:
        return len(self.__courses)

    def __add__(self, op):
        if isinstance(op, Course):
            return CourseBook(list(set([v for k, v in self.__courses.items()] + op)))
        elif isinstance(op, CourseBook):
            return CourseBook(
                list(
                    set(
                        [v for k, v in self.__courses.items()]
                        + [v for k, v in op.courses.items()]
                    )
                )
            )
        else:
            raise TypeError(
                "class <FullCourse> does not support addtion with types other than 'Course' or 'FullCourse'."
            )

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{k} : {{ length: {len(course)}, target: {'exists' if course.has_target() else 'none'} }}"
                for k, course in self.__courses.items()
            ]
        )
