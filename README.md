![Python Version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11-blue.svg)
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker-experiments-logger)](https://pypi.org/project/sagemaker-experiments-logger/)
[![Documentation](https://github.com/tsenst/lightning-experiments-logger/actions/workflows/documentation.yaml/badge.svg)](https://tsenst.github.io/lightning-experiments-logger/index.html)
# SagemakerExperimentsLogger
SagemakerExperimentsLogger provides a simple way to log experimental data such as hyperparameter settings and evaluation metrics via [AWS SageMaker Experiments API](https://aws.amazon.com/blogs/aws/amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings/). It can be easily integration into the concept of [Pytorch Lightning Trainer class](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

For detailed documentation, including the API reference, see [Read the Docs](https://tsenst.github.io/lightning-experiments-logger/index.html)

## Installation
You can install the latest (nightly) version with pip using ssh with

```bash
pip install sagemaker-experiments-logger
```

## Quickstart
The SageMaker Experiments logger can be easily applied by setup an own run context:
```Python
from pytorch_lightning import Trainer
from sagemaker.experiments.run import Run
from experiments_addon.logger import SagemakerExperimentsLogger

with Run(experiment_name="testExperiment", run_name="testRun1"):
    logger = SagemakerExperimentsLogger()
    trainer = Trainer(
        logger=logger,
        ...
    )
    ...
```
or by using an existing run context. For example in a SageMaker Training Step
```Python
from pytorch_lightning import Trainer
from experiments_addon.logger import SagemakerExperimentsLogger

logger = SagemakerExperimentsLogger(experiment_name="TestExp", run_name="TestRun")
trainer = Trainer(
logger=logger,
    ...
)
...
```

## Usage 

Try [Tutorial Notebook](https://github.com/tsenst/lightning-experiments-logger/blob/main/example/tutorial.ipynb) to learn more about the usage of the [SagemakerExperimentsLogger](https://tsenst.github.io/lightning-experiments-logger/api_logger.html) class. 

It is also worth to read the blog post: [Experiment Tracking With AWS SageMaker and PyTorch Lightning](https://medium.com/idealo-tech-blog/experiment-tracking-with-aws-sagemaker-and-pytorch-lightning-68b22fd4deee) 

## Contributing
I welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull requests are welcome.

## License
This library is licensed under the Apache 2.0 License.
