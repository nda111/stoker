"""Appending a YAML record per event, for a metric log that grows."""
from pathlib import Path
from typing import Any, Callable

import yaml

from ignite import distributed as idist
from ignite.engine import Engine


class YamlLogger:
    """Appends one YAML document per call. An ignite handler.

    Attach it to ``Events.EPOCH_COMPLETED`` for a metric log that can be read
    while the run is still going, and parsed afterwards with
    :func:`yaml.safe_load_all`.

        yaml_logger = YamlLogger(
            experiment.metrics,
            output_transform=lambda engine: {
                'epoch': engine.state.epoch,
                'training': dict(engine.state.metrics),
                'validation': dict(evaluator.state.metrics),
            },
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, yaml_logger)

    Records are separated by ``---``, so the file is a document stream rather
    than one mapping. Appending means a resumed run adds to the existing file
    instead of replacing it.

    Args:
        filepath: Where to append. Parent directories are created.
        output_transform: Builds the record from the engine. Return plain
            containers; tensors are not converted for you.

    Note:
        Writing happens on rank 0 only, so this can be attached on every rank.
        The transform still runs everywhere, which is what lets it call
        collective operations if it needs to.
    """

    def __init__(
        self,
        filepath: str | Path,
        output_transform: Callable[[Engine], dict[str, Any]],
    ):
        self.filepath = Path(filepath)
        self.output_transform = output_transform

        if idist.get_rank() == 0:
            self.filepath.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def __call__(self, engine: Engine):
        """Append one record. Writes on rank 0 only."""
        record = self.output_transform(engine)

        if idist.get_rank() != 0:
            return

        with self.filepath.open("a") as f:
            yaml.safe_dump(
                record,
                f,
                explicit_start=True,
                sort_keys=False,
            )