"""Command line configuration from type annotations.

A config is a class decorated with :func:`argconfig` that inherits
:class:`ConfigMixin`. The decorator turns it into a dataclass; the mixin reads
the field annotations and builds an :mod:`argparse` parser from them.

    @argconfig(prog='python -m train', description='Train')
    class Config(ConfigMixin):
        name: Annotated[str, Argument('name')]

        epochs: int = 100
        model: str = 'resnet18'

        @argconfig
        class OptimizerConfig(ConfigMixin):
            learning_rate: float = 1e-3

        optimizer: OptimizerConfig = default()

    config = Config.parse_args()
    print(config.optimizer.learning_rate)

The annotation decides the parser entry. A field with a default becomes an
optional flag carrying that default, a field without one becomes required, and
an ``X | None`` field defaults to ``None``. Container and choice annotations are
read as well: ``Literal`` becomes ``choices``, ``list[T]`` and ``set[T]`` become
``nargs='+'``, a fixed ``tuple[T, T]`` becomes ``nargs=2``, an :class:`~enum.Enum`
subclass becomes ``choices`` over its members, and ``bool`` becomes a
``--flag / --no-flag`` pair.

Wrap a field in :class:`Argument` to override any of that. Whatever you pass
there wins over inference, so :class:`Argument` is both the escape hatch and
the way to declare positionals and short flags.

Nested configs are flattened into a single parser rather than subparsers. A
field of config type contributes its own fields under a dotted prefix, so
``optimizer.learning_rate`` is set by ``--optimizer.learning-rate``. Note that
only the leaf name is hyphenated; the prefix keeps its underscores. Nesting may
go to any depth.

Two rules are worth stating because breaking them fails quietly rather than
loudly. A config class must inherit :class:`ConfigMixin`, since that is how a
nested field is recognised as a config at all; a nested class that omits the
mixin is treated as an ordinary scalar field and argparse will try to call it
on a string. And a mutable default must be written ``= default()``, never
``= []`` or ``= {}``.

Unsupported annotations, which raise at parser build time or feed argparse
something it cannot use: ``dict`` fields (``default()`` can build an empty one,
but no command line syntax parses into it, so treat them as default only),
unions of more than one concrete type such as ``int | str | None``, a
``Literal`` mixing value types, and heterogeneous tuples.
"""
from __future__ import annotations

import argparse
import sys
import types

from abc import ABC
from dataclasses import (
    MISSING,
    asdict,
    dataclass,
    field,
    fields,
)
from enum import Enum
from io import StringIO
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Never,
    Self,
    TextIO,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import rich
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

try:
    from omegaconf import OmegaConf

    _OMEGACONF_INSTALLED_ = True
except ImportError:
    _OMEGACONF_INSTALLED_ = False


_T_CONFIG = TypeVar('_T_CONFIG')


# ========================================================
# Argument
# ========================================================

class Argument:
    """Parser options for one field, carried inside :data:`~typing.Annotated`.

    The instance holds no logic. It is read when the parser is built, and
    anything it carries takes precedence over what would be inferred from the
    annotation.

        verbose: Annotated[bool, Argument('--verbose', '-v')] = False
        workers: Annotated[int, Argument('--workers', '-j', metavar='N')] = 4

    Args:
        *flags: Flag strings, passed to ``add_argument`` as given. Defaults to
            the hyphenated field name, so ``batch_size`` yields
            ``--batch-size``. A single flag with no leading dash declares a
            positional argument instead.
        **kwargs: Passed to ``add_argument`` verbatim, overriding any inferred
            ``type``, ``choices``, ``nargs``, ``default`` or ``required``.

    Four placements raise :exc:`TypeError` when the parser is built:

    * more than one :class:`Argument` on a single field,
    * an :class:`Argument` on a nested config field,
    * a positional outside the root config, or one whose flag does not equal
      the field name exactly (the hyphenated form is not accepted here),
    * a short flag inside a nested config, which has no room for the dotted
      prefix.
    """

    def __init__(
        self,
        *flags: str,
        **kwargs,
    ):
        self.flags = flags
        self.kwargs = kwargs


# ========================================================
# Default
# ========================================================

class _Default:
    __slots__ = ()

    def __repr__(self) -> str:
        return 'default()'


_DEFAULT = _Default()


def default() -> Never:
    """Field initializer meaning "build one from the annotation".

    Use it wherever a dataclass would reject a shared mutable default, and for
    nested configs, where it constructs the nested class so that class supplies
    its own defaults recursively.

        tags: list[str] = default()
        optimizer: OptimizerConfig = default()

    The annotation is called with no arguments, so ``int`` gives ``0``,
    ``Path`` gives ``Path('.')`` and ``list[str]`` gives ``[]``. The return
    type is :data:`~typing.Never` so that type checkers accept this on a field
    of any type; the value itself is a sentinel that :func:`argconfig` replaces
    with a real ``default_factory``. That replacement is the reason a config
    class holding ``default()`` has to be decorated. Left undecorated, the
    sentinel survives as an ordinary class attribute shared by every instance.

    Raises:
        TypeError: On an annotation it cannot construct: an optional type
            (write ``None`` yourself instead), :data:`~typing.Literal`,
            :data:`~typing.Any`, or anything that is not a class.
    """
    return _DEFAULT


def _default_factory(annotation):
    annotation, _ = _unwrap_annotated(annotation)
    annotation, optional = _unwrap_optional(annotation)

    if optional:
        raise TypeError(
            'default() cannot be used with optional types. '
            'Use None explicitly.'
        )

    origin = get_origin(annotation)

    if origin is list:
        return list

    if origin is set:
        return set

    if origin is tuple:
        return tuple

    if origin is dict:
        return dict

    if origin is Literal:
        raise TypeError(
            'default() cannot infer a default for Literal.'
        )

    if annotation is Any:
        raise TypeError(
            'default() cannot infer a default for Any.'
        )

    if isinstance(annotation, type):
        return annotation

    raise TypeError(
        f'Cannot infer a default for {annotation!r}.'
    )


def _resolve_defaults(cls):
    annotations = get_type_hints(
        cls,
        include_extras=True,
    )

    for name, annotation in annotations.items():
        value = cls.__dict__.get(
            name,
            MISSING,
        )

        if value is not _DEFAULT:
            continue

        factory = _default_factory(
            annotation
        )

        setattr(
            cls,
            name,
            field(
                default_factory=factory,
            ),
        )


# ========================================================
# ConfigMixin
# ========================================================

class ConfigMixin(ABC):
    """Parsing and export for a config class.

    Inherit this alongside the :func:`argconfig` decorator. The decorator makes
    the class a dataclass; this mixin gives it the parser, the constructors
    that read a command line, and the export and display methods.

    Inheriting it is also what marks a class as a config, so a nested config
    field is only recognised as nested when its class has this mixin.

    Despite being an :class:`~abc.ABC` it declares nothing abstract, so a
    subclass overrides nothing.
    """

    @classmethod
    def parser(
        cls,
        *args,
        **kwargs,
    ) -> argparse.ArgumentParser:
        """Build the parser for this config without running it.

        Useful to inspect the generated interface, to print help, or to add
        arguments of your own before parsing. :meth:`parse_args` calls this.

        Args:
            *args: Passed to :class:`~argparse.ArgumentParser`.
            **kwargs: Passed to :class:`~argparse.ArgumentParser`, overriding
                the ``prog``, ``description`` and ``epilog`` given to
                :func:`argconfig`.

        Returns:
            A parser with one entry per field, nested fields flattened under
            dotted prefixes.
        """
        parser_kwargs = {
            key: value
            for key, value in getattr(
                cls,
                '__parser_kwargs__',
                {},
            ).items()
            if value is not None
        }

        parser_kwargs.update(kwargs)

        parser = argparse.ArgumentParser(
            *args,
            **parser_kwargs,
        )

        _add_config_arguments(
            parser,
            cls,
        )

        return parser

    @classmethod
    def parse_args(
        cls,
        args: list[str] | None = None,
        *,
        parser_args=(),
        parser_kwargs=None,
    ) -> Self:
        """Parse a command line and return a populated config.

        The usual entry point.

            config = Config.parse_args()

        Args:
            args: Argument list. Defaults to ``sys.argv[1:]``.
            parser_args: Positional arguments for :meth:`parser`.
            parser_kwargs: Keyword arguments for :meth:`parser`.

        Returns:
            An instance of this class, with nested configs built as instances
            of their own classes.

        Raises:
            SystemExit: On a parse error or ``--help``, as argparse does.
        """
        if parser_kwargs is None:
            parser_kwargs = {}

        parser = cls.parser(
            *parser_args,
            **parser_kwargs,
        )

        namespace = parser.parse_args(args)

        return _namespace_to_config(
            cls,
            namespace,
        )

    @classmethod
    def parse_known_args(
        cls,
        args: list[str] | None = None,
        *,
        parser_args=(),
        parser_kwargs=None,
    ) -> tuple[Self, list[str]]:
        """Parse a command line, tolerating unrecognised arguments.

        Like :meth:`parse_args`, but unknown arguments are returned instead of
        causing an error. Use it when another layer, such as a launcher, takes
        the rest of the command line.

        Args:
            args: Argument list. Defaults to ``sys.argv[1:]``.
            parser_args: Positional arguments for :meth:`parser`.
            parser_kwargs: Keyword arguments for :meth:`parser`.

        Returns:
            The config, and the arguments it did not consume.
        """
        if parser_kwargs is None:
            parser_kwargs = {}

        parser = cls.parser(
            *parser_args,
            **parser_kwargs,
        )

        namespace, unknown = (
            parser.parse_known_args(args)
        )

        return (
            _namespace_to_config(
                cls,
                namespace,
            ),
            unknown,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export as a nested dict, deeply copied.

        Nested configs become nested dicts, so the result is what you want for
        a YAML dump. This is what :class:`~stoker.tracking.ConfigLogger` writes.
        """
        return asdict(self)

    def to_namespace(self) -> argparse.Namespace:
        """Export the top level fields as a namespace, shallowly.

        Nested configs are left as config objects rather than being flattened
        back into dotted keys, so this is not the inverse of parsing and will
        not round trip through :meth:`parse_args`.
        """
        return argparse.Namespace(
            **{
                dataclass_field.name: getattr(
                    self,
                    dataclass_field.name,
                )
                for dataclass_field in fields(self)
            },
        )

    def to_omegaconf(self):
        """Export as an OmegaConf ``DictConfig``.

        Raises:
            ImportError: If ``omegaconf`` is not installed. It is an optional
                dependency and nothing else here needs it.
        """
        if not _OMEGACONF_INSTALLED_:
            raise ImportError(
                'Cannot find `omegaconf` package '
                'in this environment.'
            )

        return OmegaConf.create(
            self.to_dict()
        )

    def to_tree(self) -> Tree:
        """Build the rich tree used to display this config.

        Nested configs become branches. :meth:`dump`, :meth:`__str__` and
        :meth:`__rich__` all render this.
        """
        tree = Tree(
            Text(
                type(self).__name__,
                style='bold',
            )
        )

        _build_config_tree(
            tree,
            self,
        )

        return tree

    @overload
    def dump(self):
        ...

    @overload
    def dump(self, stream: TextIO):
        ...

    def dump(
        self,
        stream: TextIO = sys.stdout,
    ):
        """Print the config as a tree.

        Args:
            stream: Where to write. Defaults to ``sys.stdout``.
        """
        rich.print(
            self,
            file=stream,
        )

    def __str__(self) -> str:
        output = StringIO()

        Console(
            file=output,
            width=120,
        ).print(
            self.to_tree(),
            end='',
        )

        return output.getvalue()

    def __rich__(self) -> Tree:
        return self.to_tree()


def _build_config_tree(
    tree: Tree,
    config: ConfigMixin,
) -> None:
    for dataclass_field in fields(config):
        name = dataclass_field.name
        value = getattr(
            config,
            name,
        )

        if isinstance(
            value,
            ConfigMixin,
        ):
            branch = tree.add(
                Text(
                    name,
                    style='bold',
                )
            )

            _build_config_tree(
                branch,
                value,
            )

            continue

        tree.add(
            Text.assemble(
                (
                    f'{name:<20}',
                    'cyan',
                ),
                repr(value),
            )
        )


# ========================================================
# Decorator
# ========================================================

@overload
def argconfig(
    cls: type[_T_CONFIG],
    /,
) -> type[_T_CONFIG]:
    ...


@overload
def argconfig(
    *,
    prog: str | None = None,
    description: str | None = None,
    epilog: str | None = None,
) -> Callable[
    [type[_T_CONFIG]],
    type[_T_CONFIG],
]:
    ...


def argconfig(
    cls=None,
    /,
    *,
    prog: str | None = None,
    description: str | None = None,
    epilog: str | None = None,
):
    """Make a class a config. Usable bare or called.

        @argconfig
        class OptimizerConfig(ConfigMixin):
            learning_rate: float = 1e-3

        @argconfig(prog='python -m train', description='Train')
        class Config(ConfigMixin):
            ...

    The class is turned into a dataclass, so the dataclass field rules apply
    unchanged. Fields without a default must come before fields with one, and
    that counts ``= default()`` and nested config fields as having defaults.
    Declaring the nested classes inside the body and listing their fields last
    is the arrangement that satisfies this.

    Decorating does not add :class:`ConfigMixin`; inherit it yourself.

    Args:
        cls: The class, when used as a bare decorator.
        prog: Program name for the parser, shown in help.
        description: Text above the argument list in help.
        epilog: Text below the argument list in help.

    Returns:
        The same class, now a dataclass, or the decorator that does this when
        called with keyword arguments.
    """

    def decorator(cls):
        # Important:
        # resolve default() before dataclass()
        _resolve_defaults(cls)

        cls = dataclass(cls)

        cls.__parser_kwargs__ = {
            'prog': prog,
            'description': description,
            'epilog': epilog,
        }

        return cls

    if cls is None:
        return decorator

    return decorator(cls)


# ========================================================
# Annotation utilities
# ========================================================

def _unwrap_annotated(annotation):
    if get_origin(annotation) is not Annotated:
        return annotation, None

    args = get_args(annotation)

    annotation = args[0]
    metadata = args[1:]

    arguments = [
        item
        for item in metadata
        if isinstance(
            item,
            Argument,
        )
    ]

    if len(arguments) > 1:
        raise TypeError(
            'Only one Argument is allowed per field.'
        )

    argument = (
        arguments[0]
        if arguments
        else None
    )

    return annotation, argument


def _unwrap_optional(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (
        Union,
        types.UnionType,
    ):
        non_none = tuple(
            arg
            for arg in args
            if arg is not type(None)
        )

        if (
            len(non_none) == 1
            and len(non_none) != len(args)
        ):
            return non_none[0], True

    return annotation, False


def _is_config_type(
    annotation,
) -> bool:
    annotation, _ = _unwrap_optional(
        annotation
    )

    return (
        isinstance(annotation, type)
        and issubclass(
            annotation,
            ConfigMixin,
        )
    )


# ========================================================
# Parser construction
# ========================================================

def _add_config_arguments(
    parser: argparse.ArgumentParser,
    cls: type[ConfigMixin],
    prefix: str = '',
) -> None:
    annotations = get_type_hints(
        cls,
        include_extras=True,
    )

    for dataclass_field in fields(cls):
        annotation = annotations[
            dataclass_field.name
        ]

        raw_annotation, argument = (
            _unwrap_annotated(
                annotation
            )
        )

        # Nested config
        if _is_config_type(
            raw_annotation
        ):
            nested_type, optional = (
                _unwrap_optional(
                    raw_annotation
                )
            )

            if optional:
                raise TypeError(
                    'Optional nested configs are '
                    'not supported.'
                )

            if argument is not None:
                raise TypeError(
                    'Argument is not allowed on '
                    'nested config fields.'
                )

            _add_config_arguments(
                parser,
                nested_type,
                prefix=(
                    f'{prefix}'
                    f'{dataclass_field.name}.'
                ),
            )

            continue

        # Regular field
        flags, argument_kwargs, _ = (
            _build_argument(
                dataclass_field,
                annotation,
            )
        )

        if prefix:
            if any(
                not flag.startswith('-')
                for flag in flags
            ):
                raise TypeError(
                    'Positional arguments are only '
                    'allowed in the root config.'
                )

            if any(
                flag.startswith('-')
                and not flag.startswith('--')
                for flag in flags
            ):
                raise TypeError(
                    'Short flags are not allowed '
                    'in nested configs.'
                )

            flags = tuple(
                (
                    f'--{prefix}{flag[2:]}'
                    if flag.startswith('--')
                    else flag
                )
                for flag in flags
            )

            argument_kwargs['dest'] = (
                f'{prefix}'
                f'{dataclass_field.name}'
            )

        parser.add_argument(
            *flags,
            **argument_kwargs,
        )


# ========================================================
# Type analysis
# ========================================================

def _analyze_type(annotation):
    annotation, _ = _unwrap_optional(
        annotation
    )

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        choices = list(args)

        if not choices:
            return (
                str,
                None,
                None,
                None,
            )

        value_types = {
            type(value)
            for value in choices
        }

        value_type = (
            next(iter(value_types))
            if len(value_types) == 1
            else str
        )

        return (
            value_type,
            choices,
            None,
            None,
        )

    if origin is list:
        element_type = (
            args[0]
            if args
            else str
        )

        return (
            element_type,
            None,
            '+',
            list,
        )

    if origin is set:
        element_type = (
            args[0]
            if args
            else str
        )

        return (
            element_type,
            None,
            '+',
            set,
        )

    if origin is tuple:
        if (
            len(args) == 2
            and args[1] is Ellipsis
        ):
            return (
                args[0],
                None,
                '+',
                tuple,
            )

        if args:
            unique_types = set(args)

            if len(unique_types) != 1:
                raise TypeError(
                    'Heterogeneous tuples are '
                    'not supported: '
                    f'{annotation!r}'
                )

            return (
                args[0],
                None,
                len(args),
                tuple,
            )

    if (
        isinstance(annotation, type)
        and issubclass(
            annotation,
            Enum,
        )
    ):
        return (
            annotation,
            list(annotation),
            None,
            None,
        )

    return (
        annotation,
        None,
        None,
        None,
    )


def _enum_parser(enum_cls):
    def parse(value):
        for member in enum_cls:
            if str(member.value) == value:
                return member

        try:
            return enum_cls[value]

        except KeyError:
            choices = ', '.join(
                str(member.value)
                for member in enum_cls
            )

            raise argparse.ArgumentTypeError(
                f'expected one of: {choices}'
            )

    return parse


def _action_takes_value(action):
    if action is None:
        return True

    if isinstance(action, str):
        return action not in {
            'store_true',
            'store_false',
            'store_const',
            'append_const',
            'count',
            'help',
            'version',
        }

    try:
        return not issubclass(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse._StoreConstAction,
                argparse._AppendConstAction,
                argparse._CountAction,
                argparse._HelpAction,
                argparse._VersionAction,
            ),
        )

    except TypeError:
        return True


def _get_default(
    dataclass_field,
):
    if (
        dataclass_field.default
        is not MISSING
    ):
        return dataclass_field.default

    if (
        dataclass_field.default_factory
        is not MISSING
    ):
        return (
            dataclass_field
            .default_factory()
        )

    return MISSING


def _build_argument(
    dataclass_field,
    annotation,
):
    annotation, argument = (
        _unwrap_annotated(
            annotation
        )
    )

    annotation, optional = (
        _unwrap_optional(
            annotation
        )
    )

    (
        inferred_type,
        inferred_choices,
        inferred_nargs,
        container,
    ) = _analyze_type(
        annotation
    )

    if argument is None:
        flags = ()
        kwargs = {}

    else:
        flags = argument.flags
        kwargs = dict(
            argument.kwargs
        )

    if not flags:
        flags = (
            '--'
            + dataclass_field.name.replace(
                '_',
                '-',
            ),
        )

    is_positional = (
        len(flags) == 1
        and not flags[0].startswith('-')
    )

    if is_positional:
        if (
            flags[0]
            != dataclass_field.name
        ):
            raise TypeError(
                'Positional argument name must '
                'match the field name: '
                f'{dataclass_field.name!r}'
            )

    else:
        kwargs.setdefault(
            'dest',
            dataclass_field.name,
        )

    default_value = _get_default(
        dataclass_field
    )

    action = kwargs.get(
        'action'
    )

    if (
        annotation is bool
        and 'action' not in kwargs
    ):
        kwargs['action'] = (
            argparse.BooleanOptionalAction
        )

        if default_value is not MISSING:
            kwargs.setdefault(
                'default',
                default_value,
            )

        elif not is_positional:
            kwargs.setdefault(
                'required',
                True,
            )

        return (
            flags,
            kwargs,
            container,
        )

    if not _action_takes_value(action):
        if default_value is not MISSING:
            kwargs.setdefault(
                'default',
                default_value,
            )

        return (
            flags,
            kwargs,
            container,
        )

    if 'type' not in kwargs:
        if (
            isinstance(
                inferred_type,
                type,
            )
            and issubclass(
                inferred_type,
                Enum,
            )
        ):
            kwargs['type'] = (
                _enum_parser(
                    inferred_type
                )
            )

        elif inferred_type is not Any:
            kwargs['type'] = (
                inferred_type
            )

    if (
        'choices' not in kwargs
        and inferred_choices is not None
    ):
        kwargs['choices'] = (
            inferred_choices
        )

    if (
        'nargs' not in kwargs
        and inferred_nargs is not None
    ):
        kwargs['nargs'] = (
            inferred_nargs
        )

    if default_value is not MISSING:
        kwargs.setdefault(
            'default',
            default_value,
        )

    elif optional:
        kwargs.setdefault(
            'default',
            None,
        )

    elif not is_positional:
        kwargs.setdefault(
            'required',
            True,
        )

    return (
        flags,
        kwargs,
        container,
    )


# ========================================================
# Namespace -> Config
# ========================================================

def _namespace_to_config(
    cls,
    namespace,
    prefix: str = '',
):
    annotations = get_type_hints(
        cls,
        include_extras=True,
    )

    values = vars(namespace)
    converted = {}

    for dataclass_field in fields(cls):
        name = dataclass_field.name
        annotation = annotations[name]

        raw_annotation, _ = (
            _unwrap_annotated(
                annotation
            )
        )

        # Nested config
        if _is_config_type(
            raw_annotation
        ):
            nested_type, optional = (
                _unwrap_optional(
                    raw_annotation
                )
            )

            if optional:
                raise TypeError(
                    'Optional nested configs are '
                    'not supported.'
                )

            converted[name] = (
                _namespace_to_config(
                    nested_type,
                    namespace,
                    prefix=f'{prefix}{name}.',
                )
            )

            continue

        # Regular field
        annotation, _ = (
            _unwrap_optional(
                raw_annotation
            )
        )

        _, _, _, container = (
            _analyze_type(
                annotation
            )
        )

        value = values[
            f'{prefix}{name}'
        ]

        if (
            value is not None
            and container is tuple
        ):
            value = tuple(value)

        elif (
            value is not None
            and container is set
        ):
            value = set(value)

        converted[name] = value

    return cls(**converted)


if __name__ == '__main__':
    from enum import StrEnum
    from pathlib import Path
    from typing import Annotated, Literal

    class Precision(StrEnum):
        FP32 = 'fp32'
        FP16 = 'fp16'
        BF16 = 'bf16'

    @argconfig(
        prog='python -m example',
        description='Train a model',
    )
    class Config(ConfigMixin):
        # ----------------------------------------
        # Nested configs
        # ----------------------------------------

        @argconfig
        class Optimizer(ConfigMixin):
            name: Literal[
                'adamw',
                'adam',
                'sgd',
            ] = 'adamw'

            learning_rate: float = 1e-3
            weight_decay: float = 1e-2

            @argconfig
            class Scheduler(ConfigMixin):
                warmup_ratio: float = 0.05
                end_ratio: float = 1e-5

                @argconfig
                class Decay(ConfigMixin):
                    power: float = 1.0

                decay: Decay = default()

            scheduler: Scheduler = default()

        # ----------------------------------------
        # Positional
        # ----------------------------------------

        name: Annotated[
            str,
            Argument('name'),
        ]

        # ----------------------------------------
        # Required
        # ----------------------------------------

        dataset: str

        # ----------------------------------------
        # Plain defaults
        # ----------------------------------------

        model: str = 'vit_tiny'
        epochs: int = 100

        # ----------------------------------------
        # Type defaults
        # ----------------------------------------

        seed: int = default()

        # ----------------------------------------
        # Argument metadata
        # ----------------------------------------

        batch_size: Annotated[
            int,
            Argument(
                '-b',
                '--batch-size',
                metavar='N',
                help='Batch size',
            ),
        ] = 256

        # ----------------------------------------
        # Enum
        # ----------------------------------------

        precision: Precision = Precision.FP32

        # ----------------------------------------
        # Optional
        # ----------------------------------------

        checkpoint: Path | None = None

        # ----------------------------------------
        # Containers
        # ----------------------------------------

        devices: list[int] | None = None

        image_size: Annotated[
            tuple[int, int],
            Argument(
                '--image-size',
                nargs=2,
                metavar=('H', 'W'),
            ),
        ] = (224, 224)

        # ----------------------------------------
        # BooleanOptionalAction
        # ----------------------------------------

        compile: bool = False

        # ----------------------------------------
        # Explicit action
        # ----------------------------------------

        verbose: Annotated[
            int,
            Argument(
                '-v',
                '--verbose',
                action='count',
                help='Increase verbosity',
            ),
        ] = 0

        # ----------------------------------------
        # Nested config defaults
        # ----------------------------------------

        optimizer: Optimizer = default()

    config = Config.parse_args()
    config.dump()
