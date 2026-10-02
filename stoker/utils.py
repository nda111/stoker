"""Helpers that did not warrant a module of their own.

Note:
    Importing this pulls in ignite, and therefore torch, because
    :func:`graceful_run` wraps an ignite engine. Keep anything cheap and
    unrelated out of here, or split it off, so that a small helper does not
    carry that cost.
"""
from collections.abc import Callable, Iterable

import rich
from rich.console import RenderableType

from ignite.engine import Engine, State

from .progress import RichProgressBar


def graceful_run(
    engine: Engine,
    data: Iterable | None = None,
    max_epochs: int | None = None,
    max_iters: int | None = None,
    epoch_length: int | None = None,
    handlers: dict[type[BaseException], RenderableType | Callable[[], None]] | None = None,
) -> State:
    """Run an engine, turning chosen exceptions into a message and a return.

    Wraps :meth:`~ignite.engine.Engine.run` so that an interruption ends the
    run deliberately instead of unwinding through a traceback. The motivating
    case is Ctrl-C during training, where the handlers already attached to the
    engine have written a checkpoint and the traceback adds nothing.

        graceful_run(
            trainer,
            data=train_loader,
            max_epochs=config.epochs,
            handlers={
                KeyboardInterrupt: 'Interrupted by user. Terminating...',
            },
        )

    Args:
        engine: The engine to run.
        data: Iterable of batches.
        max_epochs: Epochs to run.
        max_iters: Iterations to run, as an alternative to ``max_epochs``.
        epoch_length: Iterations per epoch, when ``data`` has no length.
        handlers: What to do per exception type. A string or other renderable
            is printed; a callable is called with no arguments. Omit this, or
            raise an exception not listed, to let it propagate.

    Returns:
        The engine state, whether the run completed or was handled. After a
        handled exception this is the state as of the interruption, so read
        ``state.epoch`` rather than assuming the run finished.

    Note:
        Lookup is by exact type, not :func:`isinstance`, so a subclass of a
        listed exception is not caught. A message is printed through the
        progress bar when the engine has one attached, which is what keeps it
        from landing in the middle of a bar being redrawn.
    """
    try:
        return engine.run(
            data=data,
            max_epochs=max_epochs,
            max_iters=max_iters,
            epoch_length=epoch_length,
        )

    except BaseException as e:
        if handlers is None:
            raise

        handler = handlers.get(type(e))

        if handler is None:
            raise

        if callable(handler):
            handler()

        elif hasattr(engine, 'progress_bar'):
            bar: RichProgressBar = engine.progress_bar
            bar.print(handler)

        else:
            rich.get_console().print(handler)

        return engine.state