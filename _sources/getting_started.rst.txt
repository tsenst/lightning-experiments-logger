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
    trainer.fit(...)

A more detailed demonstration of the usage of the :class:`~experiments_addon.logger.SagemakerExperimentsLogger` class can be found in the notebook: `example/tutorial.ipynb`_.

.. _example/tutorial.ipynb : https://github.com/tsenst/lightning-experiments-logger/blob/main/example/tutorial.ipynb
