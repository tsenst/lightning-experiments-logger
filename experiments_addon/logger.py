import logging
from typing import Dict, Union, Any, Optional, Callable, Iterable
from argparse import Namespace
from sagemaker.session import Session
from sagemaker.experiments import load_run
from sagemaker.experiments.run import Run
from pytorch_lightning.loggers.logger import Logger
from lightning_fabric.utilities.logger import _convert_params

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor

log = logging.getLogger(__name__)


class SagemakerExperimentsLogger(Logger):
    r"""
    Generates a summary of all layers in a :class:`~pytorch_lightning.core.module.LightningModule`.

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import ModelSummary
        >>> trainer = Trainer(callbacks=[ModelSummary(max_depth=1)])
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        super().__init__()
        self._sagemaker_session: Session = sagemaker_session
        self._disable_logging: bool = True
        self._experiment_name: Union[str, None] = experiment_name
        self._run_name: Union[str, None] = run_name
        self._name: str = ""
        self._version: str = ""
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
                ) as sagemaker_run:
                    fn(self, sagemaker_run, *args, **kwargs)
                    sagemaker_run.close()

        return log_fun

    @_sagemaker_run
    def log_hyperparams(
        self, sagemaker_run: Run, params: Union[Dict[str, Any], Namespace]
    ) -> None:
        params = _convert_params(params)
        sagemaker_run.log_parameters(params)
        sagemaker_run.close()

    @_sagemaker_run
    def log_metrics(
        self,
        sagemaker_run: Run,
        metrics: Dict[str, Union[Tensor, float]],
        step: Optional[int] = None,
    ) -> None:
        for metric_name, value in metrics.items():
            metric_value = value.item() if isinstance(value, Tensor) else value
            sagemaker_run.log_metric(
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
        sagemaker_run.log_precision_recall(
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
        sagemaker_run: Run,
        y_true: Iterable,
        y_score: Iterable,
        title: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        sagemaker_run.log_roc_curve(
            y_true=y_true, y_score=y_score, title=title, is_output=is_output
        )

    @_sagemaker_run
    def log_confusion_matrix(
        self,
        sagemaker_run: Run,
        y_true: Iterable,
        y_pred: Iterable,
        title: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        sagemaker_run.log_confusion_matrix(
            y_true=y_true, y_pred=y_pred, title=title, is_output=is_output
        )

    @_sagemaker_run
    def log_artifact(
        self,
        sagemaker_run: Run,
        name: str,
        value: str,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        sagemaker_run.log_artifact(
            name=name, value=value, media_type=media_type, is_output=is_output
        )

    @_sagemaker_run
    def log_file(
        self,
        sagemaker_run: Run,
        file_path: str,
        name: Optional[str] = None,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ) -> None:
        sagemaker_run.log_file(
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
