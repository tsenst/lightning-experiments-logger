from typing import Dict, List

import boto3
import pytest
from moto import mock_sagemaker

from experiments_addon.delete import (
    delete_experiment,
    delete_run_without_metric,
    delete_runs_like,
    has_metric,
)
from tests.test_logger import create_experiment_and_trial, create_run

EXPERIMENT_NAME = "MyExperimentA"


@pytest.fixture
def sagemaker_client():
    with mock_sagemaker():
        client = boto3.client("sagemaker", region_name="eu-central-1")
        yield client


@pytest.fixture
def experiments(sagemaker_client) -> None:
    create_experiment_and_trial(
        client=sagemaker_client,
        experiment_name=EXPERIMENT_NAME,
        run_name="MyFirstRunA",
    )
    create_run(
        client=sagemaker_client,
        experiment_name=EXPERIMENT_NAME,
        run_name="MyFirstRunB",
    )
    create_run(
        client=sagemaker_client,
        experiment_name=EXPERIMENT_NAME,
        run_name="TestRunB",
    )


def test_delete_experiment(experiments: None, sagemaker_client) -> None:
    delete_experiment(experiment_name=EXPERIMENT_NAME)
    assert sagemaker_client.list_experiments()["ExperimentSummaries"] == []


def test_delete_runs_like(experiments: None, sagemaker_client) -> None:
    delete_runs_like(experiment_name=EXPERIMENT_NAME, name_substr="FirstRun")
    assert (
        sagemaker_client.list_trial_components()["TrialComponentSummaries"][0][
            "TrialComponentName"
        ]
        == f"{EXPERIMENT_NAME}-TestRunB"
    )


def test_delete_run_without_metric(
    experiments: None, sagemaker_client, mocker
) -> None:
    mock_has_metric = mocker.patch("experiments_addon.delete.has_metric")
    mock_has_metric.side_effect = [False, True, False]

    delete_run_without_metric(experiment_name=EXPERIMENT_NAME)
    assert (
        sagemaker_client.list_trial_components()["TrialComponentSummaries"][0][
            "TrialComponentName"
        ]
        == f"{EXPERIMENT_NAME}-MyFirstRunB"
    )
    assert (
        len(sagemaker_client.list_trial_components()["TrialComponentSummaries"])
        == 1
    )
