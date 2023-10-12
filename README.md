#  ENIAT

[![License: MIT](https://img.shields.io/badge/License-MIT-azure.svg)](https://opensource.org/licenses/MIT)

Eniat is a Python template for various ML packages including PyTorch, Scikit-learn.

Currently supports only PyTorch :moyai:

It provides several convenient features

* automated training: you don't have to re-write same epoch-wise loop everytime!
* automated distributed learning for PyTorch.
* Easy configurations for your experiments.
* Various package support (in the future)

## Quick Start

### 1. Installation

Install eniat with the below command.

```bash
pip install eniat
```

eniat installation doesn't convey ML packages such as [torch](https://pytorch.org/) or [sklearn](https://scikit-learn.org/stable/), so install one you need before using eniat. :moyai:

### 2. CLI execution

You can initiate any tasks with only console commands. :moyai:

With below command, you can conduct ML experiemnts on console provided that you have proper resources for models and data. These configurations except basic arguments are based on [omegaconf](https://omegaconf.readthedocs.io/en/2.3_branch/), so refer to the documentation and provide valid arguments.

```bash
eniat -p PACKAGE -t TASK --OTHER_CONFIGS_YOU_NEED
```

Or you can simply write every configurations you need into one yaml and just type the path of it like below. I strongly recommend this, because ML experiments usually require quite large amount of parameters. Make a base configuration yaml file, and modify slightly by typing additional configurations into console command, and it will be much more convenient.

```bash
eniat -c=PATH_TO_CONFIG_FILE
```

### 3. Custom code execution

eniat is designed for various environments from mere `.py` codes to jupyter notebooks.

The most fundamental component of eniat is `Trainer`. It is in charge of training your models. In the other hand, `Grader` only evaluates your model. Below is an example code of torch training wih eniat.

```python
from eniat.utils.statelogger import DummyLogger
from eniat.torch import TorchTrainer, TorchGrader

logger = DummyLogger() # Default logger for Eniat
grader = TorchGrader(grader_cfg, logger = logger) # If none, train without evaluation
trainer = TorchTrainer(trainer_cfg, learner_cfg, data_cfg, logger, grader)

trainer.fit(device = 0) # training on dev 0
```
