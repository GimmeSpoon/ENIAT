from dataclasses import dataclass

@dataclass
class Loss:
    name:str

@dataclass
class Optimizer:
    lr:float

@dataclass
class Scheduler:
    name:str

@dataclass
class Model:
    name:str