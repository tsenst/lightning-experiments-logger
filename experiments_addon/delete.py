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
Sagemaker Experiment Delete
---------------------------
"""
import logging
import time
from typing import List

import boto3
from sagemaker.experiments._api_types import TrialComponentMetricSummary
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.session import Session

log = logging.getLogger(__name__)
def has_metric(
    metrics: List[TrialComponentMetricSummary], metric_name: str
) -> bool:
    """Checks if Metric Summaries contain the searched metric summary

    Args:
        metrics (list of TrialComponentMetricSummary):  List of trial component metrics.
        metric_name (str): Metric name to look at.

    Returns:
        Boolean that is true if metric name is in trial component metrics summary
    """
    for m in metrics:
        if m.metric_name == metric_name:
            return True
    return False


def delete_run_without_metric(
    experiment_name: str, metric_name: str = "val_loss"
) -> None:
    """Delete all runs for a given experiment i.e. trial components that do not
    have the given metric stored.

    Function can be useful to delete all experiment run that have been started but failed
    or have been aborted. E.g. due to error in the code or early detected problems with the model.

    Args:
        experiment_name (str): Determines the experiment where the run to delete are.
        metric_name (str): Determines the metric by name to check for availability.
    """
    experiment = Experiment.load(experiment_name=experiment_name)
    for trial_summary in experiment.list_trials():
        trial = _Trial.load(trial_name=trial_summary.trial_name)
        for trial_component_summary in trial.list_trial_components():
            tc = _TrialComponent.load(
                trial_component_name=trial_component_summary.trial_component_name
            )
            if not has_metric(metrics=tc.metrics, metric_name=metric_name):
                tc_name = tc.trial_component_name
                log.info(
                    f"[CLEAN UP] {tc_name} -> cause has not validation metric"
                )
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                    log.info(f"[OK] {tc_name}")
                except:
                    # tc is associated with another trial
                    log.info(f"[ERROR] {tc_name} unable to delete")
                    continue
                # to prevent throttling
                time.sleep(0.5)


def delete_runs_like(experiment_name: str, name_substr: str) -> None:
    """Delete all runs of an experiment that fulfill <name_substr in run_name>

    Function can be useful to delete several runs with similar names from the experimentation board.
    The function delete all run with run names that contain the name_substr.

    Args:
        experiment_name (str): Determines the experiment where the run to delete are.
        name_substr (str): If this substring is in the job name. The job will be deleted.
    """
    experiment_to_cleanup = Experiment.load(experiment_name=experiment_name)
    for trial_summary in experiment_to_cleanup.list_trials():
        trial = _Trial.load(trial_name=trial_summary.trial_name)
        for trial_component_summary in trial.list_trial_components():
            tc = _TrialComponent.load(
                trial_component_name=trial_component_summary.trial_component_name
            )
            tc_name = tc.trial_component_name
            if name_substr in tc_name:
                log.info(f"[CLEAN UP] {tc_name}")
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                    log.info(f"[OK] {tc_name}")
                except:
                    # tc is associated with another trial
                    log.info(f"[ERROR] {tc_name} unable to delete")
                    continue
                    # to prevent throttling
                time.sleep(0.5)


def delete_experiment(experiment_name: str) -> None:
    """
    Delete experiment and associated runs.

    Args:
        experiment_name (str): Determines the experiment to delete.

    :param str experiment_name: Experiment to delete.
    """
    session = boto3.Session()
    sm_session = Session(boto_session=session)
    log.info("[Delete] Experiment ", experiment_name)
    exp = Experiment.load(
        experiment_name=experiment_name, sagemaker_session=sm_session
    )
    exp._delete_all(action="--force")
    log.info("[OK]")
