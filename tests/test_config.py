"""Annotation handling, nesting, and the config file layer."""
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
import yaml

from stoker.config import (
    Argument,
    ConfigMixin,
    argconfig,
    default,
)


# ------------------------------------------------------------------ fixtures

@argconfig
class Nested(ConfigMixin):
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2


@argconfig
class Deeper(ConfigMixin):
    power: float = 1.0


@argconfig
class Middle(ConfigMixin):
    warmup_ratio: float = 0.05
    decay: Deeper = default()


class Precision(StrEnum):
    FP32 = 'fp32'
    FP16 = 'fp16'


@argconfig(prog='test', description='A config covering every annotation form')
class Full(ConfigMixin):
    name: Annotated[str, Argument('name')]
    required_flag: Annotated[int, Argument('--required-flag')]

    epochs: int = 100
    batch_size: int = 512
    ratio: float = 0.5
    use_amp: bool = False
    verbose: Annotated[bool, Argument('--verbose', '-v')] = False
    counted: Annotated[int, Argument('--counted', action='count')] = 0

    mode: Literal['train', 'eval'] = 'train'
    precision: Precision = Precision.FP32

    checkpoint: Path | None = None
    devices: list[int] | None = None
    tags: list[str] = default()
    image_size: Annotated[tuple[int, int], Argument('--image-size', nargs=2)] = (32, 32)

    optimizer: Nested = default()
    scheduler: Middle = default()


def write(tmp_path: Path, data: Any, name: str = 'c.yaml') -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


BASE = ['run', '--required-flag', '1']


# ------------------------------------------------------------------ defaults

def test_defaults_come_from_the_class():
    config = Full.parse_args(BASE)

    assert config.name == 'run'
    assert config.epochs == 100
    assert config.ratio == 0.5
    assert config.use_amp is False
    assert config.mode == 'train'
    assert config.precision is Precision.FP32
    assert config.checkpoint is None
    assert config.devices is None


def test_default_builds_from_the_annotation():
    config = Full.parse_args(BASE)

    assert config.tags == []
    assert isinstance(config.optimizer, Nested)
    assert config.optimizer.learning_rate == 1e-3


def test_default_is_not_shared_between_instances():
    first = Full.parse_args(BASE)
    second = Full.parse_args(BASE)

    first.tags.append('mutated')

    assert second.tags == []
    assert first.optimizer is not second.optimizer


def test_default_rejects_what_it_cannot_build():
    for annotation in [Literal['a', 'b'], Any, int | None]:
        with pytest.raises(TypeError):

            @argconfig
            class Bad(ConfigMixin):
                field: annotation = default()


# ------------------------------------------------------------------ annotations

def test_scalar_types_are_converted():
    config = Full.parse_args(BASE + ['--epochs', '7', '--ratio', '0.25'])

    assert config.epochs == 7
    assert isinstance(config.epochs, int)
    assert config.ratio == 0.25


def test_bool_becomes_a_negatable_pair():
    assert Full.parse_args(BASE + ['--use-amp']).use_amp is True
    assert Full.parse_args(BASE + ['--no-use-amp']).use_amp is False


def test_argument_overrides_inference():
    # The short flag only exists because Argument declared it.
    assert Full.parse_args(BASE + ['-v']).verbose is True
    assert Full.parse_args(BASE + ['--counted', '--counted']).counted == 2


def test_literal_restricts_choices():
    assert Full.parse_args(BASE + ['--mode', 'eval']).mode == 'eval'

    with pytest.raises(SystemExit):
        Full.parse_args(BASE + ['--mode', 'nope'])


def test_enum_accepts_its_value():
    config = Full.parse_args(BASE + ['--precision', 'fp16'])

    assert config.precision is Precision.FP16

    with pytest.raises(SystemExit):
        Full.parse_args(BASE + ['--precision', 'int4'])


def test_optional_defaults_to_none_and_converts_when_given():
    config = Full.parse_args(BASE + ['--checkpoint', 'ckpt/last.pt'])

    assert config.checkpoint == Path('ckpt/last.pt')


def test_list_takes_many_and_converts_each():
    config = Full.parse_args(BASE + ['--devices', '0', '1', '2'])

    assert config.devices == [0, 1, 2]


def test_fixed_tuple_keeps_its_length_and_type():
    config = Full.parse_args(BASE + ['--image-size', '64', '48'])

    assert config.image_size == (64, 48)
    assert isinstance(config.image_size, tuple)

    with pytest.raises(SystemExit):
        Full.parse_args(BASE + ['--image-size', '64'])


def test_a_field_without_a_default_is_required():
    with pytest.raises(SystemExit):
        Full.parse_args(['run'])


# ------------------------------------------------------------------ nesting

def test_nested_fields_use_dotted_flags():
    config = Full.parse_args(
        BASE + ['--optimizer.learning-rate', '5e-4']
    )

    assert config.optimizer.learning_rate == 5e-4
    assert config.optimizer.weight_decay == 1e-2


def test_nesting_goes_deeper_than_one_level():
    config = Full.parse_args(BASE + ['--scheduler.decay.power', '2.0'])

    assert config.scheduler.decay.power == 2.0


def test_the_leaf_is_hyphenated_and_the_prefix_is_not():
    flags = {
        flag
        for action in Full.parser()._actions
        for flag in action.option_strings
    }

    assert '--optimizer.learning-rate' in flags
    assert '--scheduler.decay.power' in flags


def test_a_nested_config_must_inherit_the_mixin():
    # Without the mixin it is treated as a scalar, so argparse calls it on a
    # string. This documents the current behaviour, which is a silent trap.
    @argconfig
    class NoMixin:
        value: int = 1

    @argconfig
    class Outer(ConfigMixin):
        inner: NoMixin = default()

    flags = {
        flag
        for action in Outer.parser()._actions
        for flag in action.option_strings
    }

    assert '--inner' in flags
    assert '--inner.value' not in flags


# ------------------------------------------------------------------ export

def test_to_dict_is_nested_and_detached():
    config = Full.parse_args(BASE)
    exported = config.to_dict()

    assert exported['optimizer']['learning_rate'] == 1e-3

    exported['tags'].append('mutated')

    assert config.tags == []


def test_str_renders_a_tree():
    text = str(Full.parse_args(BASE))

    assert 'Full' in text
    assert 'optimizer' in text
    assert 'learning_rate' in text


# ------------------------------------------------------------------ config file

def test_config_file_supplies_defaults(tmp_path):
    path = write(tmp_path, {'epochs': 7, 'optimizer': {'learning_rate': 3e-4}})

    config = Full.parse_args(BASE + ['--config', str(path)])

    assert config.epochs == 7
    assert config.optimizer.learning_rate == 3e-4
    assert config.optimizer.weight_decay == 1e-2


def test_command_line_overrides_the_file(tmp_path):
    path = write(tmp_path, {'epochs': 7})

    config = Full.parse_args(BASE + ['--config', str(path), '--epochs', '999'])

    assert config.epochs == 999


def test_file_satisfies_a_required_option(tmp_path):
    path = write(tmp_path, {'required_flag': 5})

    config = Full.parse_args(['run', '--config', str(path)])

    assert config.required_flag == 5


def test_file_satisfies_a_required_positional(tmp_path):
    path = write(tmp_path, {'name': 'from-file', 'required_flag': 1})

    config = Full.parse_args(['--config', str(path)])

    assert config.name == 'from-file'


def test_command_line_overrides_a_positional_from_the_file(tmp_path):
    path = write(tmp_path, {'name': 'from-file', 'required_flag': 1})

    config = Full.parse_args(['from-cli', '--config', str(path)])

    assert config.name == 'from-cli'


def test_a_dumped_config_reads_back_unchanged(tmp_path):
    # Dumped the way a run dumps it. ConfigLogger is the supported path because
    # importing it is what registers the representers for the field types a
    # config holds, such as enum members; plain yaml.safe_dump rejects those.
    from stoker.tracking import ConfigLogger

    original = Full.parse_args(
        BASE
        + [
            '--epochs', '3',
            '--precision', 'fp16',
            '--use-amp',
            '--devices', '0', '1',
            '--image-size', '16', '24',
            '--optimizer.learning-rate', '5e-4',
            '--scheduler.decay.power', '3.0',
        ]
    )

    path = tmp_path / 'config.yaml'
    ConfigLogger(path, params=original.to_dict())(None)

    restored = Full.parse_args(['--config', str(path)])

    assert restored == original


def test_an_empty_file_changes_nothing(tmp_path):
    path = tmp_path / 'empty.yaml'
    path.write_text('')

    assert Full.parse_args(BASE + ['--config', str(path)]).epochs == 100


def test_unknown_keys_are_an_error(tmp_path):
    path = write(tmp_path, {'nope': 1})

    with pytest.raises(KeyError, match='nope'):
        Full.parse_args(BASE + ['--config', str(path)])


def test_unknown_nested_keys_are_an_error(tmp_path):
    path = write(tmp_path, {'optimizer': {'nope': 1}})

    with pytest.raises(KeyError, match=r'optimizer\.nope'):
        Full.parse_args(BASE + ['--config', str(path)])


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Full.parse_args(BASE + ['--config', str(tmp_path / 'absent.yaml')])


def test_a_file_that_is_not_a_mapping_is_an_error(tmp_path):
    path = tmp_path / 'list.yaml'
    path.write_text('- 1\n- 2\n')

    with pytest.raises(TypeError):
        Full.parse_args(BASE + ['--config', str(path)])


def test_the_config_flag_can_be_turned_off():
    flags = {
        flag
        for action in Full.parser(config_flag=None)._actions
        for flag in action.option_strings
    }

    assert '--config' not in flags


# ------------------------------------------------------------------ parse_known_args

def test_parse_known_args_returns_the_leftovers():
    config, unknown = Full.parse_known_args(BASE + ['--not-mine', 'x'])

    assert config.name == 'run'
    assert '--not-mine' in unknown
