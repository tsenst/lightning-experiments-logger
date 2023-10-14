import os

from pytorch_lightning import Trainer

from example.model import MNISTModel
from tests.test_logger import sagemaker_session, sme_logger


def test_model_function(sme_logger, mocker):
    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    mnist_model = MNISTModel(data_dir=PATH_DATASETS)
    mocker.patch("sagemaker.experiments._metrics._MetricsManager.log_metric")

    trainer = Trainer(
        logger=sme_logger[0],
        accelerator="auto",
        limit_val_batches=1,
        limit_test_batches=1,
        limit_train_batches=1,
        devices=1,
        max_epochs=1,
    )
    trainer.fit(mnist_model)
    trainer.test()
