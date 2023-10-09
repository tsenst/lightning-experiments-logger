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
"""
SagemakerExperimentsLogger
--------------------------
"""
import logging
from argparse import Namespace
from typing import Any, Callable, Dict, Iterable, Optional, Union

from lightning_fabric.utilities.logger import _convert_params
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sagemaker.experiments import load_run
from sagemaker.experiments.run import Run
from sagemaker.session import Session
from torch import Tensor

log = logging.getLogger(__name__)


class SagemakerExperimentsLogger(Logger):
    r"""
    Log to `AWS Sagemaker Experiments <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html#run>`_ .

    Implemented using :module:`~sagemaker.experiments` API. Install api with pip:

    .. code-block:: bash

        pip install sagemaker

    It can be used in several ways:

    1. Use ``SagemakerExperimentsLogger`` by explicitly passing in ``run_name`` and ``experiment_name``.

    If ``run_name`` and ``experiment_name`` are passed in, they are honored over
    the default experiment config of the loffer. A new experiment with run job will be created.

    Note:
        Both ``run_name`` and ``experiment_name`` should be supplied to make this usage work.
        Otherwise, you may get a ``ValueError``.

    .. code:: python

        from pytorch_lightning import Trainer
        from experiments_addon.logger import SagemakerExperimentsLogger

        logger = SagemakerExperimentsLogger(experiment_name="test_experiment", run_name="test_run")
        trainer = Trainer(logger=logger)

    2. Use the ``SagemakerExperimentsLogger`` in a job script without supplying ``run_name`` and ``experiment_name``.

    In this case, the default experiment config (specified when creating the job) is fetched
    from the job environment to load the run. For example inside a training job.

    .. code:: python

        from pytorch_lightning import Trainer
        from sagemaker.experiments.run import Run:
        from experiments_addon.logger import SagemakerExperimentsLogger

        logger = SagemakerExperimentsLogger()
        trainer = Trainer(logger=logger)

    3. Use the ``SagemakerExperimentsLogger`` in a notebook within a run context (i.e. the ``with`` block)
    but without supplying ``run_name`` and ``experiment_name``.

    Every time we call ``with Run(...) as _:``, the initialized run is tracked
    in the run context and an experiment and job is created.
    Then when using ``SagemakerExperimentsLogger`` under this in the context is loaded by default.

    .. code:: python

        from pytorch_lightning import Trainer
        from sagemaker.experiments.run import Run
        from experiments_addon.logger import SagemakerExperimentsLogger

        with Run(experiment_name="test_experiment", run_name="test_run") as _:
            logger = SagemakerExperimentsLogger()
            trainer = Trainer(logger=logger)

    Args:
        run_name (str): The name of the run to be created (default: None).
            If it is None, run name of the job will be fetched to load the run.
        experiment_name (str): The name of the experiment to be created (default: None).
            Note: the experiment_name must be supplied along with a valid run_name.
            Otherwise, it will be ignored. If it is None, the experiment name will
            be fetched to load the experiment.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, one is created using the
            default AWS configuration chain.

    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        super().__init__()
        self._sagemaker_session: Session = sagemaker_session
        self._disable_logging: bool = True
        self._experiment_name: Union[str, None] = experiment_name
        self._run_name: Union[str, None] = run_name
        self._name: str = ""
        self._version: str = ""
        self._sagemaker_run: Union[Run, None] = None
        try:
            if experiment_name and run_name:
                self._name = experiment_name
                self._version = run_name
            else:
                with load_run(
                    sagemaker_session=self._sagemaker_session
                ) as sagemaker_run:
                    self._name = sagemaker_run.experiment_name
                    self._version = sagemaker_run.run_name
                    self._run_name = None
                    self._experiment_name = None
            self._disable_logging = False

        except RuntimeError as e:
            log.warning(
                f"Disable SagemakerExperimentsLogger. No current run context "
                f"has been found ({e}). To create a  sagemaker.experiments.run"
                f" explicit use experiment_name and run_name argument. "
            )

    def _sagemaker_run(fn: Callable) -> Callable:
        @rank_zero_only
        def log_fun(self, *args, **kwargs):
            if not self._disable_logging:
                with load_run(
                    experiment_name=self._experiment_name,
                    run_name=self._run_name,
                    sagemaker_session=self._sagemaker_session,
                ) as self._sagemaker_run:
                    fn(self, *args, **kwargs)
                    self._sagemaker_run.close()

        return log_fun

    @_sagemaker_run
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self._sagemaker_run.log_parameters(params)
        self._sagemaker_run.close()

    @_sagemaker_run
    def log_metrics(
        self,
        metrics: Dict[str, Union[Tensor, float]],
        step: Optional[int] = None,
    ) -> None:
        for metric_name, value in metrics.items():
            metric_value = value.item() if isinstance(value, Tensor) else value
            self._sagemaker_run.log_metric(
                name=metric_name, value=metric_value, step=step
            )

    @_sagemaker_run
    def log_precision_recall(
        self,
        sagemaker_run: Run,
        y_true: Iterable,
        predicted_probabilities: Iterable,
        positive_label: Optional[Iterable] = None,
        title: Optional[str] = None,
        is_output: bool = True,
        no_skill: Optional[int] = None,
    ) -> None:
        """Create and log a precision recall graph artifact for Sagemaker Studio UI to render.

        Function map to :func:`~sagemaker.experiments.Run.log_precision_recall
        of the `SageMaker Experiments API <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html>`_

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            predicted_probabilities (list or array): Estimated/predicted probabilities.
            positive_label (str or int): Label of the positive class (default: None).
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
            no_skill (int): The precision threshold under which the classifier cannot discriminate
                between the classes and would predict a random class or a constant class in
                all cases (default: None).

        Example::

            self.logger.log_precision_recall(...)

        """
        self._sagemaker_run.log_precision_recall(
            y_true=y_true,
            predicted_probabilities=predicted_probabilities,
            positive_label=positive_label,
            title=title,
            is_output=is_output,
            no_skill=no_skill,
        )

    @_sagemaker_run
    def log_roc_curve(
        self,
        y_true: Iterable,
        y_score: Iterable,
        title: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        """Create and log a receiver operating characteristic (ROC curve) artifact.

        Function map to :func:`~sagemaker.experiments.Run.log_roc_curve
        of the `SageMaker Experiments API <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html>`_

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_score (list or array): Estimated/predicted probabilities.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.

        Example::

            self.logger.log_roc_curve(...)

        """
        self._sagemaker_run.log_roc_curve(
            y_true=y_true, y_score=y_score, title=title, is_output=is_output
        )

    @_sagemaker_run
    def log_confusion_matrix(
        self,
        y_true: Iterable,
        y_pred: Iterable,
        title: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        """Create and log a confusion matrix artifact.

        Function map to :func:`~sagemaker.experiments.Run.log_confusion_matrix
        of the `SageMaker Experiments API <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html>`_

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_pred (list or array): Predicted labels.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.

        Example::

            self.logger.log_confusion_matrix(...)

        """
        self._sagemaker_run.log_confusion_matrix(
            y_true=y_true, y_pred=y_pred, title=title, is_output=is_output
        )

    @_sagemaker_run
    def log_artifact(
        self,
        name: str,
        value: str,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        """Record a single artifact.

        Function map to :func:`~sagemaker.experiments.Run.log_artifact
        of the `SageMaker Experiments API <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html>`_

        Args:
            name (str): The name of the artifact.
            value (str): The value.
            media_type (str): The MediaType (MIME type) of the value (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.

        Example::

            self.logger.log_artifact(...)

        """
        self._sagemaker_run.log_artifact(
            name=name, value=value, media_type=media_type, is_output=is_output
        )

    @_sagemaker_run
    def log_file(
        self,
        file_path: str,
        name: Optional[str] = None,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        """Upload a file to s3 and store it as an input/output artifact.

        Function map to :func:`~sagemaker.experiments.Run.log_file
        of the `SageMaker Experiments API <https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html>`_

        Args:
            file_path (str): The path of the local file to upload.
            name (str): The name of the artifact (default: None).
            media_type (str): The MediaType (MIME type) of the file.
                If not specified, this library will attempt to infer the media type
                from the file extension of ``file_path``.
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.

        Example::

            self.logger.log_file(...)

        """
        self._sagemaker_run.log_file(
            file_path=file_path,
            name=name,
            media_type=media_type,
            is_output=is_output,
        )

    @property
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._experiment_name

    @property
    def version(self) -> Union[int, str]:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        return self._run_name
