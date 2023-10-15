from typing import Dict, List

import boto3
import pytest
from moto import mock_sagemaker
from sagemaker.session import Session

from experiments_addon.delete import (
    delete_experiment,
    delete_run_without_metric,
    delete_runs_like,
    has_metric,
)

EXPERIMENT_NAME = "MyExperimentA"


def create_run(client, experiment_name: str, run_name: str) -> None:
    client.create_trial(
        ExperimentName=experiment_name,
        TrialName=f"{experiment_name}-{run_name}",
    )
    client.create_trial_component(
        TrialComponentName=f"{experiment_name}-{run_name}"
    )
    client.associate_trial_component(
        TrialComponentName=f"{experiment_name}-{run_name}",
        TrialName=f"{experiment_name}-{run_name}",
    )


def create_experiment_and_trial(
    client, experiment_name: str, run_name: str
) -> None:
    # The follow lines of code are a hot fix since moto seam to have a
    # problems with github action runs
    client.create_experiment(ExperimentName=experiment_name)
    client.create_trial(
        ExperimentName=experiment_name,
        TrialName=f"Default-Run-Group-{experiment_name}",
    )
    create_run(
        client=client, experiment_name=experiment_name, run_name=run_name
    )


@pytest.fixture
def sagemaker_client_session():
    with mock_sagemaker():
        client = boto3.client("sagemaker", region_name="eu-central-1")
        session = boto3.Session(region_name="eu-central-1")
        sm_session = Session(boto_session=session)
        yield client, sm_session


@pytest.fixture
def experiments(sagemaker_client_session) -> None:
    create_experiment_and_trial(
        client=sagemaker_client_session[0],
        experiment_name=EXPERIMENT_NAME,
        run_name="MyFirstRunA",
    )
    create_run(
        client=sagemaker_client_session[0],
        experiment_name=EXPERIMENT_NAME,
        run_name="MyFirstRunB",
    )
    create_run(
        client=sagemaker_client_session[0],
        experiment_name=EXPERIMENT_NAME,
        run_name="TestRunB",
    )


def test_delete_experiment(experiments: None, sagemaker_client_session) -> None:
    delete_experiment(
        experiment_name=EXPERIMENT_NAME, sagemaker_session=sagemaker_client_session[1]
    )
    assert sagemaker_client_session[0].list_experiments()["ExperimentSummaries"] == []


def test_delete_runs_like(experiments: None, sagemaker_client_session) -> None:
    delete_runs_like(
        experiment_name=EXPERIMENT_NAME,
        name_substr="FirstRun",
        sagemaker_session=sagemaker_client_session[1],
    )
    assert (
            sagemaker_client_session[0].list_trial_components()["TrialComponentSummaries"][
            0
        ]["TrialComponentName"]
            == f"{EXPERIMENT_NAME}-TestRunB"
    )


def test_delete_run_without_metric(
    experiments: None, sagemaker_client_session, mocker
) -> None:
    mock_has_metric = mocker.patch("experiments_addon.delete.has_metric")
    mock_has_metric.side_effect = [False, True, False]

    delete_run_without_metric(
        experiment_name=EXPERIMENT_NAME, sagemaker_session=sagemaker_client_session[1]
    )
    assert (
            sagemaker_client_session[0].list_trial_components()["TrialComponentSummaries"][
            0
        ]["TrialComponentName"]
            == f"{EXPERIMENT_NAME}-MyFirstRunB"
    )
    assert (
        len(
            sagemaker_client_session[0].list_trial_components()[
                "TrialComponentSummaries"
            ]
        )
        == 1
    )
