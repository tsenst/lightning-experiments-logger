from pytorch_lightning import Trainer
from experiments_addon.logger import SagemakerExperimentsLogger
from sagemaker.session import Session
from sagemaker.experiments.run import Run
import boto3
from example.model import MNISTModel
import os
def main():
    # Init our model
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    mnist_model = MNISTModel(
        data_dir=PATH_DATASETS
    )

    # sm_session = Session(boto3.session.Session(region_name="eu-central-1"))
    sm_session = None
    experiment_name = "test"
    run_name = "test9"
    # Initialize a trainer
    logger = SagemakerExperimentsLogger(
        sagemaker_session=sm_session
    )

    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        devices=1,
        max_epochs=3,
        enable_model_summary=True,
    )

    logger.log_hyperparams(params={"test" : "A"})

    # Train the model ⚡
    trainer.fit(mnist_model)
    trainer.test()

with Run(experiment_name="test", run_name="test4b", run_display_name="testa4") as _:
    main()