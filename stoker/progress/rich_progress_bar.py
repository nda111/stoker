"""Progress display for ignite engines, built on rich.

One bar per concurrent activity: an epoch counter, the training loop, and a
validation or testing loop. Attaching wires them to engine events, so the
caller only adds whatever text should ride along with each bar.

    progress_bar = RichProgressBar.Factory.create_fancy_instance()
    progress_bar.attach(training=trainer, validation=evaluator)

    # inside the train step
    progress_bar.set_postfix({'loss': f'{loss:.4f}'}, where='training')

Every method is a no op off rank 0, so this can be left in place unchanged in
a multi process run.

Attaching also sets ``engine.progress_bar``, which is how
:func:`~stoker.utils.graceful_run` finds the bar to print its message
through. Printing that way keeps the message from landing in the middle of a
bar being redrawn, and :meth:`RichProgressBar.print` is the method to use for
anything else written during a run, in place of :func:`print`.
"""
from __future__ import annotations

from typing import Any, Mapping, Literal

from rich.style import Style
from rich.progress import (
    Progress,
    ProgressColumn,
    TextColumn,
)
from rich.markdown import JustifyMethod
from rich.console import OverflowMethod

from ignite.engine import Engine, Events
from ignite import distributed as idist

from .fancy_columns import (
    FancyColumn,
    FancyBarColumn,
    FancyPercentageColumn,
)


TaskName = Literal[
    'epoch',
    'training',
    'validation',
    'testing',
]
"""Which bar a call refers to. Also the ``kind`` that selects a column theme."""


class RichProgressBar:
    """A set of rich bars driven by ignite events.

    Use :class:`Factory` rather than the constructor unless you are choosing
    your own columns. See :meth:`attach` for what gets wired up, and
    :meth:`set_postfix` for the text beside a bar.

    Bars are addressed by name, not by handle: pass ``where='training'`` and so
    on. The epoch and training bars are created when the trainer starts; the
    validation and testing bars are created on their first run and reset on
    later ones, so a repeated evaluation reuses one line instead of adding a
    line per epoch.
    """

    class Factory:
        """Preset constructions of :class:`RichProgressBar`."""

        @staticmethod
        def create_fancy_instance() -> RichProgressBar:
            """A bar with the themed columns from :mod:`.fancy_columns`.

            Icon, label and counter, then the bar, then a percentage, then the
            postfix. Each kind of task gets its own colour.
            """
            return RichProgressBar(
                FancyColumn(),
                FancyBarColumn(),
                FancyPercentageColumn(),
                TextColumn('[dim italic]{task.fields[postfix]}'),
            )

    def __init__(
        self,
        *columns: ProgressColumn,
    ):
        """Build a bar from rich columns.

        Args:
            *columns: Columns for the underlying
                :class:`~rich.progress.Progress`. Tasks here always carry a
                ``kind`` and a ``postfix`` field, so a column may read
                ``task.fields['kind']`` to vary by activity and must tolerate
                both fields being present.
        """
        self.progress = Progress(*columns)

        self.epoch_task_id = None
        self.training_task_id = None
        self.validation_task_id = None
        self.testing_task_id = None

    def attach(
        self,
        training: Engine | None = None,
        validation: Engine | None = None,
        testing: Engine | None = None,
        epoch: bool = True,
    ) -> 'RichProgressBar':
        """Wire the bars to engine events. Returns self, so it chains.

        The training engine drives the display's lifetime: the bars appear when
        it starts and the display stops when it completes or raises. Evaluation
        engines only advance their own bar.

        Args:
            training: The training engine. Drives the training bar, the epoch
                bar, and starting and stopping the display.
            validation: An engine whose runs advance the validation bar.
            testing: The same for the testing bar.
            epoch: Show the epoch counter. Needs ``training``.

        Returns:
            This instance.

        Note:
            Each engine passed here gets a ``progress_bar`` attribute pointing
            back at this object, which is what
            :func:`~stoker.utils.graceful_run` looks for. Attaching the same
            bar to engines of the same role twice is not supported, since the
            bars are keyed by role rather than by engine.
        """
        if idist.get_rank() != 0:
            return self

        if training is not None:
            self._attach_training(
                training,
                epoch=epoch,
            )

        if validation is not None:
            self._attach_evaluation(
                validation,
                where='validation',
            )

        if testing is not None:
            self._attach_evaluation(
                testing,
                where='testing',
            )

        return self

    def _attach_training(
        self,
        engine: Engine,
        epoch: bool,
    ):
        engine.add_event_handler(
            Events.STARTED,
            self._training_started,
            epoch,
        )
        engine.add_event_handler(
            Events.EPOCH_STARTED,
            self._training_epoch_started,
        )
        engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            self._training_iteration_completed,
        )
        engine.add_event_handler(
            Events.COMPLETED,
            self._training_completed,
        )

        if epoch:
            engine.add_event_handler(
                Events.EPOCH_COMPLETED,
                self._epoch_completed,
            )
            
        engine.add_event_handler(
            Events.EXCEPTION_RAISED,
            self._exception_raised,
        )

        setattr(engine, 'progress_bar', self)

    def _attach_evaluation(
        self,
        engine: Engine,
        where: Literal['validation', 'testing'],
    ):
        engine.add_event_handler(
            Events.STARTED,
            self._evaluation_started,
            where,
        )
        engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            self._evaluation_iteration_completed,
            where,
        )

        setattr(engine, 'progress_bar', self)

    def _training_started(
        self,
        engine: Engine,
        epoch: bool,
    ):
        if epoch:
            self.epoch_task_id = self.progress.add_task(
                'Epoch',
                total=engine.state.max_epochs,
                completed=engine.state.epoch,
                postfix='',
                kind='epoch',
            )

        self.training_task_id = self.progress.add_task(
            'Training',
            total=engine.state.epoch_length,
            postfix='',
            kind='training',
        )

        self.progress.start()

    def _training_epoch_started(
        self,
        engine: Engine,
    ):
        if self.training_task_id is not None:
            self.progress.reset(
                self.training_task_id,
                total=engine.state.epoch_length,
            )

    def _training_iteration_completed(
        self,
        engine: Engine,
    ):
        if self.training_task_id is not None:
            self.progress.advance(
                self.training_task_id,
            )

    def _epoch_completed(
        self,
        engine: Engine,
    ):
        if self.epoch_task_id is not None:
            self.progress.advance(
                self.epoch_task_id,
            )

    def _training_completed(
        self,
        engine: Engine,
    ):
        self.progress.stop()

    def _evaluation_started(
        self,
        engine: Engine,
        where: Literal['validation', 'testing'],
    ):
        task_id = self._get_task_id(where)

        if task_id is None:
            task_id = self.progress.add_task(
                where.capitalize(),
                total=engine.state.epoch_length,
                postfix='',
                kind=where,
            )

            self._set_task_id(
                where,
                task_id,
            )

        else:
            self.progress.reset(
                task_id,
                total=engine.state.epoch_length,
            )

    def _evaluation_iteration_completed(
        self,
        engine: Engine,
        where: Literal['validation', 'testing'],
    ):
        task_id = self._get_task_id(where)

        if task_id is not None:
            self.progress.advance(
                task_id,
            )
            
    def _exception_raised(
        self,
        engine: Engine,
        exception: BaseException,
    ):
        self.progress.stop()
        raise exception

    def _get_task_id(
        self,
        where: TaskName,
    ):
        return getattr(
            self,
            f'{where}_task_id',
        )

    def _set_task_id(
        self,
        where: TaskName,
        task_id,
    ):
        setattr(
            self,
            f'{where}_task_id',
            task_id,
        )

    def print(
        self,
        *objects: Any,
        sep: str = ' ',
        end: str = '\n',
        style: str | Style | None = None,
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        no_wrap: bool | None = None,
        emoji: bool | None = None,
        markup: bool | None = None,
        highlight: bool | None = None,
        width: int | None = None,
        height: int | None = None,
        crop: bool = True,
        soft_wrap: bool | None = None,
        new_line_start: bool = False,
    ):
        """Print above the bars, without disturbing them.

        Use this instead of :func:`print` while a run is in progress. Arguments
        are those of :meth:`rich.console.Console.print`.
        """
        if idist.get_rank() == 0:
            self.progress.console.print(
                *objects,
                sep=sep,
                end=end,
                style=style,
                justify=justify,
                overflow=overflow,
                no_wrap=no_wrap,
                emoji=emoji,
                markup=markup,
                highlight=highlight,
                width=width,
                height=height,
                crop=crop,
                soft_wrap=soft_wrap,
                new_line_start=new_line_start,
            )
    
    def set_postfix(
        self,
        mapping: Mapping,
        where: TaskName,
    ):
        """Set the text beside one bar, from a mapping.

        Each pair becomes a ``key: value`` line, so several metrics occupy
        several lines. Values are formatted as given; format them yourself to
        keep the width steady.

            bar.set_postfix({'loss': f'{loss:.4f}'}, where='training')

        Args:
            mapping: Labels to values.
            where: Which bar. Setting one that does not exist yet does nothing.
        """
        if idist.get_rank() != 0:
            return

        postfix = '\n'.join(
            f'{key}: {value}'
            for key, value in mapping.items()
        )

        task_id = self._get_task_id(where)

        if task_id is not None:
            self.progress.update(
                task_id,
                postfix=postfix,
            )
    
    def set_postfix_str(
        self,
        postfix: str,
        where: TaskName,
    ):
        """Set the text beside one bar, already formatted.

        Args:
            postfix: The text. Newlines make it span lines.
            where: Which bar. Setting one that does not exist yet does nothing.
        """
        if idist.get_rank() != 0:
            return

        task_id = self._get_task_id(where)

        if task_id is not None:
            self.progress.update(
                task_id,
                postfix=postfix,
            )
