from typing import Literal, Union, Callable, TypeVar, Sequence
import numpy as np
import pickle as pk

T_co = TypeVar('T_co', covariant=True)

def batch_load(path:Union[str, Sequence[str]], type:Literal['csv', 'npy', 'pkl']):
    if isinstance(path, str):
        if type == 'csv':
            return np.loadtxt(path, delimiter=',')
        if type == 'npy':
            return np.load(path)
        if type == 'pkl':
            return np.load(path, allow_pickle=True)
    else:
        raise NotImplementedError("sorry")

class Course:
    def __init__(self, label:str, data:T_co, target:T_co=None, transform:Callable = None) -> None:
        self.label = label
        self.data = data
        self.target = target
        self.transform = transform

        if target and (t:=len(target)) != (d:=len(data)):
            raise ValueError(f"The data has different size({d}) with the target({t}).")
        
    def has_target(self) -> bool:
        return True if self.target else False

    def __len__ (self) -> int:
        return len(self.data)
    
    def __getitem__ (self, idx:int) -> T_co:
        if self.transform:
            x = self.transform(self.data)
        if self.target:
            return x, self.target
        else:
            return x
        
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Course):
            return self.label == __value.label
        else:
            raise TypeError("class <Course> does not support comaprison between other types than 'Course'.")

class FullCourse():
    def __init__(self, courses:Union[Course, Sequence[Course]]=None, labels:Union[str, Sequence[str]]=None, datas:T_co=None, targets:T_co=None, transforms:Union[Callable, Sequence[Callable]]=None) -> None:
        
        if courses:
            if isinstance(courses, Course):
                self.courses = {courses.name : courses}
            else:
                self.courses = {course.name : course for course in courses}
        else:
            self.courses = {}

        if transforms:
            if callable(transforms):
                transforms = [transforms * len(labels)]
        
        if labels:
            if isinstance(labels, str):
                labels = [labels]
            if (l:=len(labels)) != len(datas) or l != len(targets):
                raise ValueError(f"The size of labels, data, targets, and transforms must match.")
            for label, data, target, transform in zip(labels, datas, targets, transforms if transforms else [None * len(labels)]):
                self.courses[label] = Course(label, data, target, transform)
        
    def append(self, course:Course = None, label:str = None, data = None):
        if course:
            self.courses[course.label] = course
        elif data and label:
            self.courses[label] = data
        else:
            raise ValueError("At least one of 'course' or a pair of 'label' and 'data' must be given")
        return self
    
    def __len__(self) -> int:
        return len(self.courses)

    def __add__ (self, op):
        if isinstance(op, Course):
            return FullCourse(list(set([v for k, v in self.courses.items()] + op)))
        elif isinstance(op, FullCourse):
            return FullCourse(list(set([v for k, v in self.courses.items()] + [v for k, v in op.courses.items()])))
        else:
            raise TypeError("class <FullCourse> does not support addtion with types other than 'Course' or 'FullCourse'.")

    def __repr__(self) -> str:
        return '\n'.join([f"{k} : {{ length: {len(course)}, target: {'exists' if course.has_target() else 'none'} }}" for k, course in self.courses.items()])
