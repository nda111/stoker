"""Learning rate schedule: a linear warmup followed by cosine annealing."""
import warnings

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ignite.handlers import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)


def create_cosine_annealing_with_warmup_scheduler(
    optimizer: Optimizer,
    start_lr: float = 1.0E-8,
    peak_lr: float = 1.0E-3,
    end_lr: float = 1.0E-8,

    total_iters: int | None = None,
    num_epochs: int | None = None,
    data_loader: DataLoader | None = None,
    epoch_length: int | None = None,

    warmup_ratio: float = 0.05,
):
    """Build a warmup then cosine scheduler, given the length of the run.

    The learning rate rises linearly from ``start_lr`` to ``peak_lr`` over the
    first ``warmup_ratio`` of the run, then follows a cosine curve down to
    ``end_lr``. Attach the result to ``Events.ITERATION_COMPLETED``; it steps
    per iteration, not per epoch.

        scheduler = create_cosine_annealing_with_warmup_scheduler(
            optimizer=optimizer,
            start_lr=0,
            peak_lr=config.optimizer.learning_rate,
            end_lr=config.optimizer.learning_rate * 1e-5,
            num_epochs=config.epochs,
            data_loader=train_loader,
        )
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

    The total length must be known up front, and there are three ways to say
    it. Give ``total_iters`` directly, or give ``num_epochs`` with either a
    ``data_loader`` to measure or an ``epoch_length`` to multiply. Passing more
    than one is allowed but warns, and the more specific wins:
    ``total_iters``, then ``data_loader``, then ``epoch_length``.

    Args:
        optimizer: Whose ``lr`` is scheduled.
        start_lr: Learning rate at the first iteration.
        peak_lr: Learning rate at the end of warmup.
        end_lr: Learning rate the cosine descends to. Keep it above zero if
            the final steps should still move.
        total_iters: Total iterations in the run.
        num_epochs: Epochs in the run. Needed by both forms below.
        data_loader: Measured with :func:`len` to get the iterations per epoch.
        epoch_length: Iterations per epoch, when the loader has no length.
        warmup_ratio: Fraction of the run spent warming up. Set it to 0 for
            pure cosine annealing.

    Returns:
        An ignite scheduler, callable as an event handler.

    Raises:
        ValueError: If the length cannot be determined, that is if none of the
            three forms is complete.

    The schedule is sized so that the last iteration of the run is the one that
    reaches ``end_lr``, which means the length you declare has to be the length
    you actually run. Running longer restarts the cosine from ``peak_lr``, since
    an ignite cosine scheduler cycles rather than holding its end value.
    """
    if total_iters is not None:
        redundant = []

        if num_epochs is not None:
            redundant.append("num_epochs")
        if data_loader is not None:
            redundant.append("data_loader")
        if epoch_length is not None:
            redundant.append("epoch_length")

        if redundant:
            warnings.warn(
                f"{', '.join(redundant)} ignored because "
                f"'total_iters' is explicitly specified.",
                UserWarning,
                stacklevel=2,
            )

    elif data_loader is not None:
        if num_epochs is None:
            raise ValueError(
                "'num_epochs' is required when 'data_loader' is specified."
            )

        if epoch_length is not None:
            warnings.warn(
                "'epoch_length' ignored because 'data_loader' is specified.",
                UserWarning,
                stacklevel=2,
            )

        total_iters = num_epochs * len(data_loader)

    elif epoch_length is not None:
        if num_epochs is None:
            raise ValueError(
                "'num_epochs' is required when 'epoch_length' is specified."
            )

        total_iters = num_epochs * epoch_length

    else:
        raise ValueError(
            "Specify 'total_iters', 'data_loader', or 'epoch_length'."
        )

    warmup_length = int(total_iters * warmup_ratio)

    # The last warmup iteration and the first cosine iteration are the same
    # event, both at peak_lr, so the cosine spans one more position than the
    # iterations left to it. Without the +1 it wraps into its next cycle on the
    # final iteration and returns peak_lr there.
    cosine_length = total_iters - warmup_length

    if warmup_length > 0:
        cosine_length += 1

    scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=peak_lr,
        end_value=end_lr,
        cycle_size=cosine_length,
    )

    if warmup_length > 0:
        scheduler = create_lr_scheduler_with_warmup(
            scheduler,
            warmup_start_value=start_lr,
            warmup_duration=warmup_length,
            warmup_end_value=peak_lr,
        )

    return scheduler