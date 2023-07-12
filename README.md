#  ENIAT

[![License: MIT](https://img.shields.io/badge/License-MIT-azure.svg)](https://opensource.org/licenses/MIT)

Eniat is a Python template for various ML packages such as PyTorch or Scikit-learn

It provides several convenient features

* automated training: you don't have to re-write same epoch loop everytime! (Pytorch)
* automated distributed learning (PyTorch)
* hydra config, logging integration
* Package independent classes such as Trainer, Grader, etc.
* some remote features: logging, evaluation, etc. (<strong>in progress</strong>)

## Quick Start
### 1. CLI execution

Install eniat with below command.

```bash
pip install eniat
```

You can initiate any tasks with only console command.

With below command, it does nothing but will show you default config.
```bash
eniat
```

For training or whatever task you want, just type command in the hydra style.
You should have some knowledges about hydra configuration. Below command will initiate training based on a config file located in `./config` and named `basic.yaml`.

```
eniat -cd=./config -cn=basic task=train
```

Detailed modification of you config files is required for better configuration. I recommend to refer hydra [documentation](https://hydra.cc/docs/intro/) if you're not familiar with it.

### 2. Custom code execution

eniat is designed for various environments from mere `.py` codes to jupyter notebooks.

You would have to use [Compose API](https://hydra.cc/docs/advanced/compose_api/) for hydra config integration with the Jupyter notebook.

The most basic component of eniat is `Trainer`. It is in charge of any training task. In the other hand, `Grader` only evaluates predictions or your model. Below is an example of PyTorch training wih eniat.

```python
from eniat.utils.statelogger import StateLogger
from eniat.torch import TorchTrainer, TorchGrader

logger = StateLogger() # Default logger for Eniat
grader = TorchGrader(grader_cfg, logger = logger) # If none, train without evaluation
trainer = TorchTrainer(trainer_cfg, learner_cfg, data_cfg, logger, grader)

trainer.fit(device = 0)
```