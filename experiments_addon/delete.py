import time
from typing import List

import boto3
from sagemaker.experiments._api_types import TrialComponentMetricSummary
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.session import Session


def has_metric(
    metrics: List[TrialComponentMetricSummary], metric_name: str
) -> bool:
    """
    Checks if Metric Summaries contain the searched metric summary

    :param metrics: List of trial component metrics
    :param str metric_name: Metric name to look at
    :return: True if metric name is in trial component metrics summary
    """
    for m in metrics:
        if m.metric_name == metric_name:
            return True
    return False


def delete_jobs_without_metric(
    experiment_name: str, metric_name: str = "val_loss"
) -> None:
    """
    Delete all jobs for a given experiment i.e. trial components that do not
    have the given metric.

    :param str experiment_name: Experiment name.
    :param str metric_name: Metric name to look at.
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
                print(
                    f"[CLEAN UP] {tc_name} -> cause has not validation metric"
                )
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                    print(f"[OK] {tc_name}")
                except:
                    # tc is associated with another trial
                    print(f"[ERROR] {tc_name} unable to delete")
                    continue
                # to prevent throttling
                time.sleep(0.5)


def delete_job_like(name_substr: str, experiment_name: str) -> None:
    """
    Delete all jobs of an experiment that fulfill <name_substr in job_name>

    :param str name_substr: If this substring is in the job name. The job will
    be deleted.
    :param str experiment_name: Experiment name to look at.
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
                print(f"[CLEAN UP] {tc_name}")
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                    print(f"[OK] {tc_name}")
                except:
                    # tc is associated with another trial
                    print(f"[ERROR] {tc_name} unable to delete")
                    continue
                    # to prevent throttling
                time.sleep(0.5)


def delete_experiment(experiment_name: str) -> None:
    """
    Delete a given experiment.

    :param str experiment_name: Experiment to delete.
    """
    session = boto3.Session()
    sm_session = Session(boto_session=session)
    print("[Delete] Experiment ", experiment_name)
    exp = Experiment.load(
        experiment_name=experiment_name, sagemaker_session=sm_session
    )
    exp._delete_all(action="--force")
    print("[OK]")
