"""Metric history across epochs, with tables and checkpointing.

Division of labour with ignite: an :class:`~ignite.metrics.Metric` reduces over
the batches within one epoch, and :class:`MetricState` records one already
reduced value per epoch. It is not a :class:`~ignite.metrics.Metric` and has no
``attach``; feeding it is the caller's job, usually from an
``EPOCH_COMPLETED`` handler reading ``engine.state.metrics``.

    state = MetricState(
        training=dict(lr=[], loss=[]),
        validation=dict(loss=[], top1=[], top5=[]),
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def record(engine):
        state.update(
            epoch=engine.state.epoch,
            training=dict(lr=lr, **engine.state.metrics),
            validation=evaluator.state.metrics,
        )
        history, current = state.summary(best_key='valid.top1;big')
        console.print(history)

The structure passed to the constructor is the schema, and it is fixed from
there on. A metric given an empty list is *historical* and gains one entry per
epoch; a metric given a scalar holds only its latest value. Which one a metric
is decides where it appears in :meth:`MetricState.summary`, as a column of the
History table or a row of the State table. Metrics outside the schema are
rejected by :meth:`MetricState.update`.

Metrics are addressed by a prefixed display name, ``train.loss``,
``valid.top1``, ``test.top5``, in :meth:`MetricState.summary` arguments. The
prefixes are the short forms, so ``training.loss`` is not a valid name there.

:class:`MetricState` subclasses :class:`~ignite.base.Serializable`, so it can
go straight into the ``to_save`` mapping of an ignite
:class:`~ignite.handlers.Checkpoint` and the history is restored on resume.

Nothing here is distributed aware or device aware. In a multi process run every
rank keeps its own copy, and reducing across ranks is the job of the
:class:`~ignite.metrics.Metric` upstream, such as :class:`AverageMeter`. Store
Python scalars; a value kept as a CUDA tensor stays resident on the device for
the lifetime of the history.
"""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import torch

from ignite.base import Serializable

from rich import box
from rich.style import Style
from rich.table import Table
from rich.text import Text


# ============================================================
# MetricGroup
# ============================================================

class MetricGroup(dict):
    """One group of metrics: a dict that type checks and allows attribute access.

    This is what :attr:`MetricState.training` and its siblings are. It behaves
    as a :class:`dict`, and additionally ``group.loss`` reads and writes
    ``group['loss']``.

    Values are restricted to what can be stored and serialized: numbers,
    strings, ``None``, numpy scalars and arrays, tensors, lists and tuples.
    Anything else raises :exc:`TypeError` on assignment. Keys must be strings.

    Because every attribute assignment is redirected into the dict, instances
    hold no real attributes, and a missing key read as an attribute raises
    :exc:`AttributeError` rather than :exc:`KeyError`.

    Note:
        Assigning a key here bypasses the owning :class:`MetricState`'s schema,
        which is read only at construction. A metric added this way is invisible
        to :meth:`MetricState.update` and :meth:`MetricState.summary`, yet it is
        written by :meth:`MetricState.state_dict`, where it then fails the
        unexpected key check on load. Declare metrics in the constructor.
    """

    _allowed_value_types = (
        int,
        float,
        bool,
        str,
        type(None),
        np.generic,
        list,
        tuple,
        torch.Tensor,
        np.ndarray,
    )

    def __init__(
        self,
        metrics: Mapping[str, Any] | None = None,
    ):
        super().__init__()

        if metrics is not None:
            self.update(metrics)

    @classmethod
    def _validate(
        cls,
        key: str,
        value: Any,
    ) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f'Metric key must be str, '
                f'got {type(key).__name__}.'
            )

        if not isinstance(
            value,
            cls._allowed_value_types,
        ):
            raise TypeError(
                f'Metric \'{key}\' has unsupported type '
                f'{type(value).__name__}.'
            )

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self._validate(key, value)
        super().__setitem__(key, value)

    def __setattr__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self[key] = value

    def __getattr__(
        self,
        key: str,
    ) -> Any:
        try:
            return self[key]

        except KeyError as e:
            raise AttributeError(key) from e

    def update(
        self,
        metrics=(),
        /,
        **kwargs,
    ) -> None:
        """Merge a mapping and keyword arguments, validating each value.

        Unlike :meth:`dict.update` this goes through the type check, so it
        raises :exc:`TypeError` on an unsupported value rather than storing it.
        """
        items = dict(metrics, **kwargs)

        for key, value in items.items():
            self[key] = value

# ============================================================
# MetricState
# ============================================================

class MetricState(Serializable):
    """Metric history for the training, validation and testing groups.

    The three groups share one :attr:`epoch` list, so all of them are recorded
    by a single :meth:`update` call per epoch.

    Attributes:
        epoch: The epoch numbers recorded so far, strictly increasing.
        training: The training group, a :class:`MetricGroup`.
        validation: The validation group.
        testing: The testing group.

    Note:
        There is no way to clear the history short of constructing a new
        instance. :meth:`load_state_dict` replaces :attr:`epoch` but leaves
        metrics absent from the incoming dict untouched.
    """

    _state_keys = (
        'epoch',
        'training',
        'validation',
        'testing',
    )

    _group_keys = (
        'training',
        'validation',
        'testing',
    )

    _group_aliases = {
        'train': 'training',
        'valid': 'validation',
        'test': 'testing',
    }

    _group_prefixes = {
        'training': 'train',
        'validation': 'valid',
        'testing': 'test',
    }

    def __init__(
        self,
        training: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        testing: Mapping[str, Any] | None = None,
    ):
        """Declare the metrics. This structure is the schema from here on.

        Give a metric an empty list to make it historical, keeping one value
        per epoch, or a scalar to keep only its latest value.

            MetricState(
                training=dict(loss=[], lr=0.0),
                validation=dict(top1=[]),
            )

        Args:
            training: Metric names to initial values for the training group.
            validation: The same for validation.
            testing: The same for testing.
        """
        super().__init__()

        self.epoch: list[int] = []

        self.training = MetricGroup(training)
        self.validation = MetricGroup(validation)
        self.testing = MetricGroup(testing)

        # The initial structure defines the schema.
        self._schema = {
            key: set(getattr(self, key))
            for key in self._group_keys
        }

        # A metric initialized as a list is historical.
        self._historical = {
            key: [
                name
                for name, value in getattr(self, key).items()
                if isinstance(value, list)
            ] for key in self._group_keys
        }

    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        epoch: int,
        training: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        testing: Mapping[str, Any] | None = None,
        strict: bool = False,
        default: Any = None,
    ) -> None:
        """Record one epoch across all three groups.

        Historical metrics get a value appended, so their lists stay as long as
        :attr:`epoch`; a historical metric left out of this call is padded with
        ``default``. Non historical metrics are overwritten only when present.

        Call this once per epoch with everything you have. Epochs must strictly
        increase, so a second call for the same epoch, one after training and
        another after validation, is rejected.

        Args:
            epoch: The epoch number. Must be a real :class:`int`, greater than
                the last recorded one. A numpy integer or a single element
                tensor raises.
            training: Metric names to values for the training group. Keys must
                be in the schema.
            validation: The same for validation.
            testing: The same for testing.
            strict: Require every metric in the schema to be present.
            default: What to append for a historical metric that was not given.

        Raises:
            TypeError: If ``epoch`` is not an :class:`int`, or a group is
                neither a :class:`~collections.abc.Mapping` nor ``None``.
            ValueError: If ``epoch`` does not exceed the last one, or if
                ``strict`` is set and a metric is missing.
            KeyError: If a metric is not in the schema. The message gives the
                prefixed names.

        Note:
            Only keys are checked before anything is written. A value of an
            unsupported type is caught during the commit, by which point
            :attr:`epoch` has already been appended.
        """
        if not isinstance(epoch, int):
            raise TypeError(
                f'epoch must be int, '
                f'got {type(epoch).__name__}.'
            )

        if self.epoch and epoch <= self.epoch[-1]:
            raise ValueError(
                'epoch must be strictly increasing: '
                f'last={self.epoch[-1]}, new={epoch}.'
            )

        updates = {
            'training': training,
            'validation': validation,
            'testing': testing,
        }

        # Validate everything before modifying state.
        for group_name, values in updates.items():
            self._validate_update(
                group_name,
                values,
                strict=strict,
            )

        # Only commit epoch after validation succeeds.
        self.epoch.append(epoch)

        for group_name, values in updates.items():
            self._update_group(
                group_name,
                values,
                default=default,
            )

    def _validate_update(
        self,
        group_name: str,
        values: Mapping[str, Any] | None,
        strict: bool,
    ) -> None:
        if values is None:
            values = {}

        if not isinstance(values, Mapping):
            raise TypeError(
                f'\'{group_name}\' must be a Mapping or None, '
                f'got {type(values).__name__}.'
            )

        schema = self._schema[group_name]
        provided = set(values)

        unexpected = provided - schema

        if unexpected:
            prefix = self._group_prefixes[group_name]

            keys = [
                f'{prefix}.{key}'
                for key in sorted(unexpected)
            ]

            raise KeyError(
                f'Unexpected metrics: {keys}.'
            )

        if strict:
            missing = schema - provided

            if missing:
                prefix = self._group_prefixes[group_name]

                keys = [
                    f'{prefix}.{key}'
                    for key in sorted(missing)
                ]

                raise KeyError(
                    f'Missing metrics: {keys}.'
                )

    def _update_group(
        self,
        group_name: str,
        values: Mapping[str, Any] | None,
        default: Any,
    ) -> None:
        values = values or {}

        group = getattr(self, group_name)
        schema = self._schema[group_name]
        historical = self._historical[group_name]

        for key in schema:
            if key in historical:
                group[key].append(
                    values.get(key, default)
                )

            elif key in values:
                group[key] = values[key]

    # ========================================================
    # Summary
    # ========================================================

    def summary(
        self,
        recent: int = 3,
        best: int = 1,
        best_key: str = 'train.loss;small',
        excluded_keys: list[str] | None = None,
    ) -> tuple[Table, Table]:
        """Render the history and the current state as two rich tables.

        The History table holds the historical metrics, one column each, with
        the last ``recent`` epochs and then the best ``best`` epochs in a
        highlighted section below a separator. The State table lists the non
        historical metrics. Neither is printed; hand them to a console.

            history, current = state.summary(best_key='valid.top1;big')
            console.print(history)

        Args:
            recent: How many of the latest epochs to show.
            best: How many best epochs to append. Pass 0 to omit the section,
                which is also how to call this before ``best_key``'s metric
                exists.
            best_key: Which metric ranks the best epochs, as
                ``'<metric>;<mode>'``. The mode is ``big`` or ``small`` for
                the largest or smallest value, and ``'epoch;epoch'`` is the
                special case of ranking by epoch number. The metric is a
                prefixed name such as ``'valid.top1'`` and must be historical.
            excluded_keys: Prefixed names to leave out, plus the bare
                ``'epoch'`` to drop the epoch column.

        Returns:
            The History table and the State table.

        Raises:
            ValueError: If ``recent`` or ``best`` is negative, if ``best_key``
                is malformed, or if its metric is not historical.
            KeyError: If ``best_key``'s metric or group prefix is unknown.

        Note:
            Epochs whose ranking value is ``None``, NaN or not a scalar are
            not eligible, so fewer than ``best`` rows may appear. Those rows
            run worst first, best last.
        """
        if recent < 0:
            raise ValueError(
                'recent must be >= 0.'
            )

        if best < 0:
            raise ValueError(
                'best must be >= 0.'
            )

        excluded = set(excluded_keys or [])

        historical = self._historical_columns(
            excluded,
        )

        non_historical = self._non_historical_columns(
            excluded,
        )

        history_table = self._history_table(
            historical=historical,
            excluded=excluded,
            recent=recent,
            best=best,
            best_key=best_key,
        )

        state_table = self._state_table(
            non_historical,
        )

        return history_table, state_table

    def _history_table(
        self,
        historical: dict[str, list],
        excluded: set[str],
        recent: int,
        best: int,
        best_key: str,
    ) -> Table:
        table = Table(
            title='History',
            show_header=True,
            header_style='bold',
            box=box.SIMPLE,
            padding=(0, 1),
        )

        sort_key, sort_mode = self._parse_best_key(
            best_key,
        )

        show_epoch = 'epoch' not in excluded

        if show_epoch:
            table.add_column(
                self._column_header(
                    'epoch',
                    sort_key,
                    sort_mode,
                ),
                justify='right',
            )

        for key in historical:
            table.add_column(
                self._column_header(
                    key,
                    sort_key,
                    sort_mode,
                ),
                justify='right',
            )

        # --------------------------------------------
        # Recent epochs
        # --------------------------------------------

        n = len(self.epoch)

        recent_indices = list(
            range(
                max(0, n - recent),
                n,
            )
        )

        for index in recent_indices:
            table.add_row(
                *self._history_row(
                    index,
                    historical,
                    show_epoch,
                )
            )

        # --------------------------------------------
        # Best epochs
        # --------------------------------------------

        best_indices = self._best_indices(
            best_key=best_key,
            best=best,
        )

        if best_indices:
            if recent_indices:
                table.add_section()

            best_style = Style(
                bold=True,
                reverse=True,
            )

            for index in best_indices:
                table.add_row(
                    *self._history_row(
                        index,
                        historical,
                        show_epoch,
                    ),
                    style=best_style,
                )

        return table

    def _state_table(
        self,
        non_historical: dict[str, Any],
    ) -> Table:
        table = Table(
            title='State',
            show_header=True,
            header_style='bold',
        )

        table.add_column('metric')
        table.add_column(
            'value',
            justify='right',
        )

        for key, value in non_historical.items():
            table.add_row(
                key,
                self._format_value(value),
            )

        return table

    # ========================================================
    # Best
    # ========================================================

    def _parse_best_key(
        self,
        best_key: str,
    ) -> tuple[
        str,
        Literal['big', 'small', 'epoch'],
    ]:
        try:
            key, mode = best_key.rsplit(
                ';',
                maxsplit=1,
            )

        except ValueError as e:
            raise ValueError(
                'best_key must have the form '
                '\'<metric>;big\', '
                '\'<metric>;small\', or '
                '\'epoch;epoch\'.'
            ) from e

        if mode not in {
            'big',
            'small',
            'epoch',
        }:
            raise ValueError(
                f'Unknown best mode \'{mode}\'. '
                'Expected \'big\', \'small\', or \'epoch\'.'
            )

        if mode == 'epoch':
            if key != 'epoch':
                raise ValueError(
                    'Epoch sorting must use '
                    'best_key=\'epoch;epoch\'.'
                )

        elif key == 'epoch':
            raise ValueError(
                'Epoch sorting must use '
                'best_key=\'epoch;epoch\'.'
            )

        return key, mode

    def _best_indices(
        self,
        best_key: str,
        best: int,
    ) -> list[int]:
        if best == 0 or not self.epoch:
            return []

        key, mode = self._parse_best_key(
            best_key,
        )

        if mode == 'epoch':
            # Worst -> best:
            # smaller epoch -> larger epoch
            indices = sorted(
                range(len(self.epoch)),
                key=lambda i: self.epoch[i],
            )

            return indices[-best:]

        values = self._resolve_historical_metric(
            key,
        )

        candidates = [
            i
            for i, value in enumerate(values)
            if self._is_sortable(value)
        ]

        if mode == 'big':
            # Worst -> best:
            # smaller value -> larger value
            candidates.sort(
                key=lambda i: self._scalar(values[i]),
            )

        else:
            # Worst -> best:
            # larger value -> smaller value
            candidates.sort(
                key=lambda i: self._scalar(values[i]),
                reverse=True,
            )

        return candidates[-best:]

    # ========================================================
    # Column collection
    # ========================================================

    def _historical_columns(
        self,
        excluded: set[str],
    ) -> dict[str, list]:
        columns = {}

        for group_name in self._group_keys:
            prefix = self._group_prefixes[group_name]
            group = getattr(self, group_name)

            for key in self._historical[group_name]:
                full_key = f'{prefix}.{key}'

                if full_key in excluded:
                    continue

                columns[full_key] = group[key]

        return columns

    def _non_historical_columns(
        self,
        excluded: set[str],
    ) -> dict[str, Any]:
        columns = {}

        for group_name in self._group_keys:
            prefix = self._group_prefixes[group_name]
            group = getattr(self, group_name)

            for key in self._schema[group_name]:
                if key in self._historical[group_name]:
                    continue

                full_key = f'{prefix}.{key}'

                if full_key in excluded:
                    continue

                columns[full_key] = group[key]

        return columns

    # ========================================================
    # Key resolution
    # ========================================================

    def _resolve_historical_metric(
        self,
        key: str,
    ) -> list:
        if '.' not in key:
            raise KeyError(
                f'Invalid metric key \'{key}\'. '
                'Expected e.g. \'train.loss\'.'
            )

        alias, metric = key.split(
            '.',
            maxsplit=1,
        )

        try:
            group_name = self._group_aliases[alias]

        except KeyError as e:
            raise KeyError(
                f'Unknown metric group \'{alias}\'.'
            ) from e

        if metric not in self._schema[group_name]:
            raise KeyError(
                f'Unknown metric \'{key}\'.'
            )

        if metric not in self._historical[group_name]:
            raise ValueError(
                f'Metric \'{key}\' is not historical.'
            )

        return getattr(
            self,
            group_name,
        )[metric]

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _column_header(
        key: str,
        sort_key: str,
        sort_mode: Literal['big', 'small', 'epoch'],
    ) -> Text:
        if key != sort_key:
            return Text(key)

        arrow = (
            '↓'
            if sort_mode == 'small'
            else '↑'
        )

        return Text(
            f'{key} ({arrow})',
            style='bold reverse',
        )

    def _history_row(
        self,
        index: int,
        historical: dict[str, list],
        show_epoch: bool,
    ) -> list[str]:
        row = []

        if show_epoch:
            row.append(
                str(self.epoch[index])
            )

        for values in historical.values():
            if index < len(values):
                value = values[index]
            else:
                value = None

            row.append(
                self._format_value(value)
            )

        return row

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    'Best metric tensor must contain '
                    'exactly one element.'
                )

            return float(value.item())

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    'Best metric ndarray must contain '
                    'exactly one element.'
                )

            return float(value.item())

        if isinstance(value, np.generic):
            return float(value)

        return float(value)

    @classmethod
    def _is_sortable(
        cls,
        value: Any,
    ) -> bool:
        if value is None:
            return False

        try:
            scalar = cls._scalar(value)

        except (
            TypeError,
            ValueError,
        ):
            return False

        return not np.isnan(scalar)

    @staticmethod
    def _format_value(
        value: Any,
    ) -> str:
        if value is None:
            return '-'

        if isinstance(value, float):
            return f'{value:.6g}'

        if isinstance(value, np.floating):
            return f'{float(value):.6g}'

        if (
            isinstance(value, torch.Tensor)
            and value.numel() == 1
        ):
            return f'{value.item():.6g}'

        if (
            isinstance(value, np.ndarray)
            and value.size == 1
        ):
            return f'{value.item():.6g}'

        return str(value)

    # ========================================================
    # Serialization
    # ========================================================

    def state_dict(self) -> OrderedDict:
        """Export the history for checkpointing.

        Empty groups and an empty :attr:`epoch` are omitted, so a fresh
        instance exports an empty mapping. The result is a deep copy, so it is
        a snapshot that later :meth:`update` calls do not alter.
        """
        state = OrderedDict()

        if self.epoch:
            state['epoch'] = deepcopy(self.epoch)

        for key in self._group_keys:
            group = getattr(self, key)

            if group:
                state[key] = deepcopy(dict(group))

        return state

    def load_state_dict(
        self,
        state_dict: Mapping,
    ) -> None:
        """Restore the history, as ignite does when resuming from a checkpoint.

        This merges rather than replaces. :attr:`epoch` is overwritten, but a
        metric absent from ``state_dict`` keeps whatever it holds. Loading into
        a populated instance is therefore only safe when the incoming dict
        covers the same metrics, which it does when it came from
        :meth:`state_dict` on a matching schema.

        Args:
            state_dict: A mapping as produced by :meth:`state_dict`.

        Raises:
            TypeError: If ``state_dict`` is not a
                :class:`~collections.abc.Mapping`, or a value has the wrong
                type.
            ValueError: If it holds unexpected keys, or if the restored
                historical lists do not all have one entry per epoch.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                'state_dict must be a Mapping, '
                f'got {type(state_dict).__name__}.'
            )

        expected_keys = set(self._state_keys)
        provided_keys = set(state_dict)

        unexpected_keys = (
            provided_keys - expected_keys
        )

        if unexpected_keys:
            raise KeyError(
                'Unexpected keys: '
                f'{sorted(unexpected_keys)}.'
            )

        epoch = state_dict.get(
            'epoch',
            [],
        )

        if not isinstance(epoch, list):
            raise TypeError(
                '\'epoch\' must be a list, '
                f'got {type(epoch).__name__}.'
            )

        self.epoch = list(epoch)

        for key in self._group_keys:
            value = state_dict.get(
                key,
                {},
            )

            if not isinstance(value, Mapping):
                raise TypeError(
                    f'\'{key}\' must be a Mapping, '
                    f'got {type(value).__name__}.'
                )

            schema = self._schema[key]

            unexpected = set(value) - schema

            if unexpected:
                raise KeyError(
                    f'Unexpected metrics in \'{key}\': '
                    f'{sorted(unexpected)}.'
                )

            group = getattr(self, key)

            for metric in schema:
                if metric in value:
                    group[metric] = value[metric]

        self._validate_history_lengths()

    def _validate_history_lengths(self) -> None:
        expected = len(self.epoch)

        for group_name in self._group_keys:
            group = getattr(self, group_name)

            for key in self._historical[group_name]:
                actual = len(group[key])

                if actual != expected:
                    prefix = self._group_prefixes[
                        group_name
                    ]

                    raise ValueError(
                        'Historical metric '
                        f'\'{prefix}.{key}\' has length '
                        f'{actual}, but epoch has length '
                        f'{expected}.'
                    )

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self) -> str:
        args = []

        if self.epoch:
            args.append(
                f'epoch={self.epoch!r}'
            )

        for key in self._group_keys:
            group = getattr(self, key)

            if group:
                args.append(
                    f'{key}={dict(group)!r}'
                )

        return (
            f'{self.__class__.__name__}'
            f'({", ".join(args)})'
        )

    def __str__(self) -> str:
        groups = []

        if self.epoch:
            groups.append(
                f'epoch: {self.epoch}'
            )

        for key in self._group_keys:
            group = getattr(self, key)

            if not group:
                continue

            values = ', '.join(
                f'{name}={value}'
                for name, value in group.items()
            )

            groups.append(
                f'{key}: {values}'
            )

        return (
            '\n'.join(groups)
            if groups
            else f'{self.__class__.__name__}()'
        )
