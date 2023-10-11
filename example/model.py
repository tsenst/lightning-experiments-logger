# Copyright 2023 Tobias Senst.
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
"""
MNISTModel
----------
"""
import torch
from typing import Tuple, Optional
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics import MetricCollection
from torchvision import transforms
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
)
import time
from torchvision.datasets import MNIST


def create_metric_collection(no_classes: int) -> MetricCollection:
    metrics = {
        "Micro-Precision": MulticlassPrecision(
            num_classes=no_classes, average="micro"
        ),
        "Micro-Recall": MulticlassRecall(
            num_classes=no_classes, average="micro"
        ),
        "Macro-Precision": MulticlassPrecision(
            num_classes=no_classes, average="macro"
        ),
        "Macro-Recall": MulticlassRecall(
            num_classes=no_classes, average="macro"
        ),
        "Micro-F1": MulticlassF1Score(num_classes=no_classes, average="micro"),
        "Macro-F1": MulticlassF1Score(num_classes=no_classes, average="macro"),
        "Accuracy": MulticlassAccuracy(num_classes=no_classes),
    }

    return MetricCollection(metrics)


class MNISTModel(LightningModule):
    def __init__(
        self,
        data_dir: str,
        learning_rate: float = 0.02,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        no_classes: int = 10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.save_hyperparameters()
        self.transform = transforms.ToTensor()
        self.train_metrics = create_metric_collection(
            no_classes=no_classes
        ).clone(prefix="Train-")
        self.val_metrics = create_metric_collection(
            no_classes=no_classes
        ).clone(prefix="Val-")
        self.test_metrics = create_metric_collection(
            no_classes=no_classes
        ).clone(prefix="Test-")
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.train_loss = 0
        self.train_step_count = 0
        self.val_loss = 0
        self.val_step_count = 0
        self.tic = time.time()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: Optional[int]
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        metrics = self.train_metrics(preds, y)
        self.train_loss += loss.item()
        self.train_step_count += 1
        metrics["loss"] = self.train_loss / self.train_step_count
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_epoch_runtime = time.time() - self.tic
        self.tic = time.time()
        self.train_loss = 0
        self.train_step_count = 0
        metric_dict = self.train_metrics.compute()
        metric_dict["Runtime"] = self.train_epoch_runtime
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, logger=True)
        self.train_metrics.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: Optional[int]
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        metrics = self.val_metrics(preds, y)
        self.val_loss += loss.item()
        self.val_step_count += 1
        metrics["loss"] = self.val_loss / self.val_step_count
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=False)

    def on_validation_epoch_end(self) -> None:
        self.val_loss = 0
        self.val_step_count = 0
        metric_dict = self.val_metrics.compute()
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: Optional[int]
    ) -> None:
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_metrics.update(preds, y)

    def on_test_epoch_end(self) -> None:
        metric_dict = self.test_metrics.compute()
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True, logger=True)
        self.test_metrics.reset()

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.hparams.train_batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.hparams.val_batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.hparams.test_batch_size
        )
