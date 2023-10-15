![Python Version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11-blue.svg)
[![Unit Test](https://github.com/tsenst/lightning-experiments-logger/actions/workflows/python-package.yml/badge.svg)](https://github.com/tsenst/lightning-experiments-logger/actions/workflows/python-package.yml)
# SagemakerExperimentsLogger
SagemakerExperimentsLogger provides a simple way to log experimental data such as hyperparameter settings and evaluation metrics via [AWS SageMaker Experiments API](https://aws.amazon.com/blogs/aws/amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings/). It can be easily integration into the concept of [Pytorch Lightning Trainer class](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

## Installation
You can install the latest (nightly) version with pip using ssh with

```bash
pip install git+ssh://git@github.com:tsenst/lightning-experiments-logger.git
```

## Quickstart
```Python
from experiments_addon.logger import SagemakerExperimentsLogger

with Run(experiment_name="testExperiment", run_name="testRun1"):
    logger = SagemakerExperimentsLogger()
    trainer = Trainer(
        logger=logger,
        ...
    )
    ...
```
or
```Python
from experiments_addon.logger import SagemakerExperimentsLogger

logger = SagemakerExperimentsLogger(experiment_name="TestExp", run_name="TestRun")
trainer = Trainer(
logger=logger,
    ...
)
...
```

## Usage 
To learn more about the usage of the *SagemakerExperimentsLogger* class try out the [Tutorial Notebook](https://github.com/tsenst/lightning-experiments-logger/blob/main/example/tutorial.ipynb).

## Contributing
We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.

## License
This library is licensed under the Apache 2.0 License.
