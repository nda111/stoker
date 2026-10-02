# stoker

Scaffolding for [pytorch-ignite](https://pytorch-ignite.ai) training runs.
ignite starts the fire; this keeps it fed.

Declarative configuration, metric history, progress display and experiment
bookkeeping, as separate pieces that can be adopted one at a time.

## Install

```bash
pip install stoker
```

From a checkout:

```bash
pip install -e .
pip install -e '.[omegaconf]'   # adds ConfigMixin.to_omegaconf
```

Requires Python 3.11 or newer, for `typing.Never`, `typing.Self` and
`enum.StrEnum`. Depends on `torch`, `pytorch-ignite`, `rich`, `pyyaml`,
`numpy` and `typing_extensions`.

To reproduce the environment this was developed against, pinned down to the
patch version, use either of:

```bash
conda env create -f environment.yaml   # creates an env named stoker
conda activate stoker
pip install -e .

pip install -r requirements.txt
pip install -e .
```

Run them from the project directory. Both pin the CUDA 12.8 build of torch,
which has to match the local NVIDIA driver, so change the index URL each file
carries if yours differs. Neither installs the package itself, hence the
separate editable install.

`pyproject.toml` holds the library's own dependency ranges and is what
installing the package follows, so these two files are for development rather
than for consumers.

## Quick start

```python
from ignite.engine import Events

from stoker.config import argconfig, ConfigMixin, Annotated, Argument, default
from stoker.metrics import MetricState, AverageMeter
from stoker.progress import RichProgressBar
from stoker.schedulers import create_cosine_annealing_with_warmup_scheduler
from stoker.tracking import Experiment, ConfigLogger, YamlLogger
from stoker.utils import graceful_run


@argconfig(prog='python -m train', description='Train')
class Config(ConfigMixin):
    name: Annotated[str, Argument('name')]

    epochs: int = 100
    batch_size: int = 512

    @argconfig
    class OptimizerConfig(ConfigMixin):
        learning_rate: float = 1e-3
        weight_decay: float = 1e-2

    optimizer: OptimizerConfig = default()


config = Config.parse_args()        # --optimizer.learning-rate 3e-4
config.dump()

experiment = Experiment(name=config.name)

# Your own data, model and engines, built as ignite usually does.
train_loader, valid_loader = build_loaders(config)
model, optimizer = build_model(config)
trainer, evaluator = build_engines(model, optimizer, config)

state = MetricState(
    training=dict(lr=[], loss=[]),
    validation=dict(loss=[], top1=[]),
)

progress_bar = RichProgressBar.Factory.create_fancy_instance()
progress_bar.attach(training=trainer, validation=evaluator)

trainer.add_event_handler(
    Events.STARTED,
    ConfigLogger(experiment.config, params=config.to_dict()),
)


@trainer.on(Events.EPOCH_COMPLETED)
def record(engine):
    state.update(
        epoch=engine.state.epoch,
        training=dict(
            lr=optimizer.param_groups[0]['lr'],
            **engine.state.metrics,
        ),
        validation=evaluator.state.metrics,
    )
    history, current = state.summary(best_key='valid.top1;big')
    progress_bar.print(history)


graceful_run(
    trainer,
    data=train_loader,
    max_epochs=config.epochs,
    handlers={KeyboardInterrupt: 'Interrupted by user. Terminating...'},
)
```

`examples/train_cifar100.py` is a complete working example: a timm model on
CIFAR-100 with warmup and cosine annealing, mixed precision, checkpointing and
the progress display. It expects the package to be installed, additionally
needs `timm` and `torchvision`, and writes to `data/` and `runs/` relative to
the working directory:

```bash
pip install -e . timm torchvision
python examples/train_cifar100.py my-first-run --epochs 10
```

## What is in it

| Module | What it gives you |
| --- | --- |
| `stoker.config` | A config class becomes an `argparse` parser. Nested configs flatten into dotted flags such as `--optimizer.learning-rate`. |
| `stoker.metrics` | `MetricState` keeps one value per epoch and renders history and state tables. `AverageMeter` averages whatever the engine returns. |
| `stoker.progress` | `RichProgressBar`, one bar per activity, wired to engine events and themed per activity. |
| `stoker.tracking` | `Experiment` decides the run directory. `ConfigLogger` dumps settings once, `YamlLogger` appends a record per epoch. |
| `stoker.schedulers` | Linear warmup into cosine annealing, sized from epochs and a data loader. |
| `stoker.datasets` | Channel normalization statistics for MNIST, CIFAR-10, CIFAR-100 and ImageNet. |
| `stoker.registry` | Name to object lookup, for selecting an implementation by string. |
| `stoker.utils` | `graceful_run`, which turns chosen exceptions into a message instead of a traceback. |

## Conventions

These hold across modules, so they are stated once here rather than repeated
in every docstring.

**Distributed runs.** Everything that writes a file or draws on the terminal
acts on rank 0 and is a no op elsewhere, so the same code runs unchanged under
`idist`. `Experiment` goes further: the run id is taken on rank 0 and
broadcast, so every rank agrees on one directory. Reduction across ranks is
the job of the ignite metric upstream, not of `MetricState`, which keeps a
separate copy per rank.

**ignite handler shape.** `ConfigLogger` and `YamlLogger` are callables taking
an engine, so they attach with `add_event_handler`. `RichProgressBar` attaches
itself to several events through `attach`. Schedulers returned here attach to
`Events.ITERATION_COMPLETED`, since they step per iteration.

**Printing during a run.** Use `RichProgressBar.print`, not `print`, or output
lands in the middle of a bar being redrawn. `attach` also sets
`engine.progress_bar`, which is how `graceful_run` finds the bar to print its
message through.

**Paths.** `Experiment` owns the directory layout. Nothing else decides where
files go; ask it for `config`, `metrics`, `periodic_checkpoints` and so on.
Creating one refuses to start if the destination exists, so a finished run
cannot be written over.

**Import cost.** Submodules are independent, and `stoker/__init__.py` is empty
on purpose, so importing one part does not pay for the rest. `stoker.config`
costs about 100 ms and pulls neither ignite nor torch, which makes it usable in
a plain command line tool. `stoker.utils` pulls ignite and therefore torch.

## Things that are easy to get wrong

`stoker.datasets.stats` constants are `(std, mean)`, in that order, which is
the reverse of what `torchvision.transforms.Normalize` takes. Unpack them
rather than splatting.

`MetricState` fixes its schema at construction. A metric given an empty list
keeps one value per epoch; one given a scalar keeps only its latest value;
anything not declared there is rejected by `update`.

Metrics are addressed in `summary` by prefixed short names, `train.loss` and
`valid.top1`, not by the full group names.

The scheduler is sized so that the final iteration of the run reaches
`end_lr`. Running longer than the declared length restarts the cosine from
`peak_lr`, because an ignite cosine scheduler cycles rather than holding its
end value.

Importing `stoker.tracking.config_logger` registers a representer on
`yaml.SafeDumper` so that a `torch.device` dumps as a string. That is a
process wide change to PyYAML.

## Documentation

Every public name has a docstring, so the library documents itself:

```bash
python -m pydoc stoker.config            # module overview and full API
python -m pydoc stoker.metrics.MetricState
python -m pydoc -b                       # browse everything
```

In a notebook or IPython, `MetricState.summary?` works the same way.

## License

MIT. See [LICENSE](LICENSE).
