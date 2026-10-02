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
    To continue a run instead of beginning one, use :meth:`open`, which is a
    separate entry point precisely so that reopening cannot happen by accident.

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
        self._resume_index = 0

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

    @classmethod
    def open(
        cls,
        name: str,
        root: str | Path = 'runs',
        run_id: str = 'latest',
    ) -> 'Experiment':
        """Reopen an existing run, to continue it.

        Unlike the constructor this requires the directory to be there and does
        not create a new one, so the paths point at the earlier run's files and
        a checkpoint found under them can be loaded back.

            experiment = Experiment.open(config.name)

            checkpoint = experiment.latest_checkpoints / 'last'
            if checkpoint.exists():
                Checkpoint.load_objects(to_load, torch.load(checkpoint))

        Args:
            name: The experiment name to look under.
            root: Directory holding all experiments.
            run_id: Which run, or ``'latest'`` for the most recent by name.
                Run ids sort chronologically, being timestamps.

        Returns:
            An instance pointing at that run.

        Raises:
            FileNotFoundError: If the experiment or the run does not exist.

        Note:
            Resolving ``'latest'`` happens on rank 0 and is broadcast, so every
            rank continues the same run even if one of them sees a directory
            appear mid-resolution. :attr:`config` moves aside to a numbered
            name so that reopening does not overwrite the settings the run
            started with, while :attr:`metrics` stays the same file and is
            appended to.
        """
        self = cls.__new__(cls)
        self.name = name
        self.root = Path(root)

        resolved = (
            self._resolve_run_id(run_id)
            if idist.get_rank() == 0
            else None
        )
        self._run_id = idist.broadcast(resolved, src=0)

        index = (
            self._next_resume_index()
            if idist.get_rank() == 0
            else None
        )
        self._resume_index = idist.broadcast(index, src=0)

        if idist.get_rank() == 0:
            self.periodic_checkpoints.mkdir(parents=True, exist_ok=True)
            self.best_checkpoints.mkdir(parents=True, exist_ok=True)

        idist.barrier()

        return self

    def _resolve_run_id(self, run_id: str) -> str:
        parent = self.root / self.name

        if run_id != 'latest':
            if not (parent / run_id).is_dir():
                raise FileNotFoundError(
                    f'Run not found: {parent / run_id}'
                )
            return run_id

        if not parent.is_dir():
            raise FileNotFoundError(
                f'Experiment not found: {parent}'
            )

        runs = sorted(
            entry.name
            for entry in parent.iterdir()
            if entry.is_dir()
        )

        if not runs:
            raise FileNotFoundError(
                f'No runs under: {parent}'
            )

        return runs[-1]

    def _next_resume_index(self) -> int:
        if not (self.path / 'config.yaml').is_file():
            return 0

        existing = list(self.path.glob('config.resumed-*.yaml'))

        return len(existing) + 1

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
        """The config snapshot, written once by :class:`.ConfigLogger`.

        ``config.yaml`` for a fresh run. On a run reopened by :meth:`open` it
        becomes ``config.resumed-01.yaml``, then ``-02`` and so on, so the
        settings each attempt ran with are all kept. The number is fixed when
        the instance is created, so this does not change under you as files
        appear.
        """
        if self._resume_index == 0:
            return self.path / 'config.yaml'

        return self.path / f'config.resumed-{self._resume_index:02d}.yaml'

    @property
    def resumed(self) -> bool:
        """Whether this came from :meth:`open` on a run that had been started."""
        return self._resume_index > 0

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
