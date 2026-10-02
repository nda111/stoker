"""Where a run writes: directory layout for one experiment."""
from datetime import datetime
from pathlib import Path

import ignite.distributed as idist


class Experiment:
    """The directories for one run, named by the time it started.

    Every path a run writes to comes from here, so nothing else in the library
    decides where files go.

        experiment = Experiment(name=config.name)
        experiment.path            # runs/<name>/20260102_143000
        experiment.config          # that, plus config.yaml
        experiment.best_checkpoints

    Creating one creates the checkpoint directories, and refuses to start if
    the destination is already there, so an existing run cannot be written over.

    The run id is the timestamp, taken on rank 0 and broadcast, so all ranks
    agree on one directory even though their clocks are read at different
    moments. Only rank 0 creates anything, and the constructor ends at a
    barrier, so later ranks do not race ahead of the directories existing.

    Attributes:
        name: The experiment name, the directory under ``root``.
        root: Where experiments live.
    """

    def __init__(
        self,
        name: str,
        root: str | Path = 'runs',
    ):
        """Pick the run id and create the checkpoint directories.

        Args:
            name: Groups runs of the same experiment.
            root: Directory holding all experiments.

        Raises:
            FileExistsError: If :attr:`path` already exists.
        """
        self.name = name
        self.root = Path(root)

        run_id = (
            datetime.now().strftime('%Y%m%d_%H%M%S')
            if idist.get_rank() == 0
            else None
        )
        self._run_id = idist.broadcast(run_id, src=0)

        self.check_existence()

        if idist.get_rank() == 0:
            self.periodic_checkpoints.mkdir(
                parents=True,
                exist_ok=False,
            )
            self.best_checkpoints.mkdir(
                parents=True,
                exist_ok=False,
            )

        idist.barrier()

    @property
    def run_id(self) -> str:
        """The timestamp identifying this run, shared by all ranks."""
        return self._run_id

    @property
    def path(self) -> Path:
        """The run directory, ``root/name/run_id``."""
        return self.root / self.name / self.run_id

    @property
    def checkpoints(self) -> Path:
        """Holds the checkpoint directories, and the latest checkpoint itself."""
        return self.path / 'checkpoints'

    @property
    def periodic_checkpoints(self) -> Path:
        """For checkpoints saved every so many epochs."""
        return self.checkpoints / 'periodic'

    @property
    def best_checkpoints(self) -> Path:
        """For checkpoints kept by score."""
        return self.checkpoints / 'best'

    @property
    def latest_checkpoints(self) -> Path:
        """For the rolling latest checkpoint.

        Note:
            This is :attr:`checkpoints` itself rather than a subdirectory, so
            the latest checkpoint sits beside the ``periodic`` and ``best``
            directories instead of inside one of its own.
        """
        return self.checkpoints

    @property
    def config(self) -> Path:
        """The config snapshot, written once by :class:`.ConfigLogger`."""
        return self.path / 'config.yaml'

    @property
    def metrics(self) -> Path:
        """The per epoch metric log, appended to by :class:`.YamlLogger`."""
        return self.path / 'metrics.yaml'

    def check_existence(self) -> None:
        """Raise if :attr:`path` is taken. Called by the constructor.

        The check runs on rank 0 and the answer is broadcast, so every rank
        fails together rather than some proceeding.

        Raises:
            FileExistsError: If the run directory already exists.
        """
        exists = (
            self.path.exists()
            if idist.get_rank() == 0
            else None
        )
        exists = idist.broadcast(exists, src=0)

        if exists:
            raise FileExistsError(
                f'Experiment already exists: {self.path}'
            )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'run_id={self.run_id!r}, '
            f'path={str(self.path)!r})'
        )
