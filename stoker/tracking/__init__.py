"""Experiment bookkeeping: where a run writes, and what it records there.

:class:`Experiment` decides the directory layout, and the two loggers are
ignite handlers that write into it: :class:`ConfigLogger` dumps the settings
once at the start, :class:`YamlLogger` appends a record per epoch.

    experiment = Experiment(name=config.name)

    trainer.add_event_handler(
        Events.STARTED,
        ConfigLogger(experiment.config, params=config.to_dict()),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        YamlLogger(experiment.metrics, output_transform=record),
    )

This is bookkeeping, not :mod:`logging`: no levels, handlers or formatters, and
nothing here writes to the console.
"""
from .experiment import Experiment
from .yaml_logger import YamlLogger
from .config_logger import ConfigLogger
