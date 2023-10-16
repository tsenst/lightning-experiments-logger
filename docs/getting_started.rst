Quick Started
-------------
The SageMaker Experiments logger can be easily applied:

.. code-block:: python

    from experiments_addon.logger import SagemakerExperimentsLogger

    logger = SagemakerExperimentsLogger(experiment_name="TestExp", run_name="TestRun")
    trainer = Trainer(
    logger=logger,
    ...
    )

A more detailed demonstration of the usage of the :class:`experiments_addon.logger.SagemakerExperimentsLogger` class can be found in the `tutorial.ipynb notebook`_.

.. _tutorial.ipynb notebook: https://github.com/tsenst/lightning-experiments-logger/blob/main/example/tutorial.ipynb
