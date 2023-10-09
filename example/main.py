# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import boto3
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sagemaker.experiments.run import Run
from sagemaker.session import Session

from example.model import MNISTModel
from experiments_addon.logger import SagemakerExperimentsLogger


def main():
    # Init our model
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    mnist_model = MNISTModel(data_dir=PATH_DATASETS)

    sm_session = Session(boto3.session.Session(region_name="eu-central-1"))
    sagemaker_logger = SagemakerExperimentsLogger(sagemaker_session=sm_session)
    tensorboard_logger = TensorBoardLogger()

    trainer = Trainer(
        logger=[sagemaker_logger, tensorboard_logger],
        accelerator="auto",
        devices=1,
        max_epochs=3,
        enable_model_summary=True,
    )
    trainer.fit(mnist_model)
    trainer.test()


with Run(
    experiment_name="testExperiment", run_name="testRun3"
) as _:
    main()
