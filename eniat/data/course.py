from typing import Literal

class Course(object):
    def __init__(self, dataset, label:str=None) -> None:
        self.dataset = dataset
        self.label = label

        if len(dataset) != len(label):
            raise ValueError('The data and labels are not paired! They have different sizes.')

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx:int):
        return self.dataset[idx], self.label[idx] if self.label else self.dataset[idx]
    
class FullCourse(Course):
    def __init__(self, train:Course=None, eval:Course=None, predict:Course=None) -> None:
        self.train = train
        self.train.label = 'train'
        self.eval = eval
        self.eval.label = 'eval'
        self.predict = predict
        self.predict.label = 'predict'
        self.dataset = self.train

    def select(self, data:Literal['train', 'eval', 'predict']):
        self.dataset = getattr(self, data)
    
    def __getitem__(self, idx:int|Literal['train', 'eval', 'predict']):
        if isinstance(idx, str):
            return getattr(self, idx)
        else:
            return super().__getitem__(idx)