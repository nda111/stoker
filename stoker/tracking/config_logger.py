"""Writing the config of a run to YAML, once, at the start.

Note:
    Importing this registers a representer on :class:`yaml.SafeDumper` so that
    a :class:`torch.device` dumps as a string instead of failing. That is a
    process wide change to PyYAML, not something scoped to this class.
"""
from pathlib import Path

import yaml

from ignite import distributed as idist
from ignite.engine import Engine

try:
    import torch
    def represent_torch_device(
        dumper: yaml.SafeDumper,
        device: torch.device,
    ):
        """Dump a :class:`torch.device` as its string form."""
        return dumper.represent_scalar(
            'tag:yaml.org,2002:str',
            str(device),
        )


    yaml.SafeDumper.add_representer(
        torch.device,
        represent_torch_device,
    )
except ImportError:
    pass

class ConfigLogger:
    """Dumps a config to YAML when the engine starts. An ignite handler.

    Attach it to ``Events.STARTED`` so that the settings of a run are on disk
    before anything can fail.

        config_logger = ConfigLogger(experiment.config, params=config.to_dict())
        trainer.add_event_handler(Events.STARTED, config_logger)

    Args:
        filepath: Where to write. Its parent must exist, which it does when
            the path came from :class:`Experiment`.
        params: What to dump, as plain containers. Use
            :meth:`~stoker.config.ConfigMixin.to_dict`.

    Note:
        Key order is preserved rather than sorted, so the file reads in
        declaration order. Writing happens on rank 0 only.
    """

    def __init__(
        self,
        filepath: str | Path,
        params: dict,
    ):
        self.filepath = Path(filepath)
        self.params = params

    def __call__(self, engine: Engine):
        """Write the file. Overwrites whatever is there."""
        if idist.get_rank() != 0:
            return

        with self.filepath.open("w") as f:
            yaml.safe_dump(
                self.params,
                f,
                sort_keys=False,
            )
