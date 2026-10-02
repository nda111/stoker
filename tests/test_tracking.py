"""Experiment directories, reopening a run, and the two file loggers."""
import time

import pytest
import yaml
from ignite.engine import Engine, Events

from stoker.tracking import ConfigLogger, Experiment, YamlLogger


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'runs'


def engine_of(epochs):
    @Engine
    def engine(engine, batch):
        pass

    return engine, epochs


# ------------------------------------------------------------------ layout

def test_a_fresh_run_creates_its_directories(root):
    experiment = Experiment('demo', root=root)

    assert experiment.path.is_dir()
    assert experiment.periodic_checkpoints.is_dir()
    assert experiment.best_checkpoints.is_dir()
    assert experiment.path == root / 'demo' / experiment.run_id


def test_paths_hang_off_the_run_directory(root):
    experiment = Experiment('demo', root=root)

    assert experiment.config == experiment.path / 'config.yaml'
    assert experiment.metrics == experiment.path / 'metrics.yaml'
    assert experiment.checkpoints == experiment.path / 'checkpoints'
    assert experiment.periodic_checkpoints == experiment.checkpoints / 'periodic'
    assert experiment.best_checkpoints == experiment.checkpoints / 'best'


def test_the_latest_checkpoint_sits_beside_the_others_not_under_them(root):
    experiment = Experiment('demo', root=root)

    assert experiment.latest_checkpoints == experiment.checkpoints


def test_the_run_id_is_a_sortable_timestamp(root):
    first = Experiment('demo', root=root)
    time.sleep(1.1)
    second = Experiment('demo', root=root)

    assert first.run_id < second.run_id


def test_a_fresh_run_is_not_resumed(root):
    assert Experiment('demo', root=root).resumed is False


# ------------------------------------------------------------------ open

def test_open_reuses_the_latest_run(root):
    first = Experiment('demo', root=root)
    time.sleep(1.1)
    second = Experiment('demo', root=root)

    reopened = Experiment.open('demo', root=root)

    assert reopened.run_id == second.run_id
    assert reopened.run_id != first.run_id
    assert reopened.resumed is True


def test_open_accepts_an_explicit_run_id(root):
    first = Experiment('demo', root=root)
    time.sleep(1.1)
    Experiment('demo', root=root)

    reopened = Experiment.open('demo', root=root, run_id=first.run_id)

    assert reopened.run_id == first.run_id


def test_open_points_at_the_earlier_files(root):
    experiment = Experiment('demo', root=root)
    experiment.metrics.write_text('--- {epoch: 1}\n')

    reopened = Experiment.open('demo', root=root)

    assert reopened.metrics == experiment.metrics
    assert reopened.metrics.read_text() == '--- {epoch: 1}\n'


def test_open_creates_checkpoint_directories_that_are_missing(root):
    experiment = Experiment('demo', root=root)
    experiment.best_checkpoints.rmdir()

    reopened = Experiment.open('demo', root=root)

    assert reopened.best_checkpoints.is_dir()


def test_open_on_an_unknown_experiment_is_an_error(root):
    with pytest.raises(FileNotFoundError):
        Experiment.open('never-ran', root=root)


def test_open_on_an_unknown_run_is_an_error(root):
    Experiment('demo', root=root)

    with pytest.raises(FileNotFoundError):
        Experiment.open('demo', root=root, run_id='19990101_000000')


def test_open_with_no_runs_at_all_is_an_error(root):
    (root / 'demo').mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        Experiment.open('demo', root=root)


# ------------------------------------------------------------------ resumed config

def test_a_reopened_run_moves_the_config_aside(root):
    experiment = Experiment('demo', root=root)
    experiment.config.write_text('epochs: 10\n')

    first = Experiment.open('demo', root=root)
    first.config.write_text('epochs: 20\n')
    second = Experiment.open('demo', root=root)

    assert first.config.name == 'config.resumed-01.yaml'
    assert second.config.name == 'config.resumed-02.yaml'
    assert experiment.config.read_text() == 'epochs: 10\n'


def test_the_resumed_number_is_zero_padded(root):
    experiment = Experiment('demo', root=root)
    experiment.config.write_text('a: 1\n')

    for index in range(1, 4):
        reopened = Experiment.open('demo', root=root)
        reopened.config.write_text('a: 1\n')

        assert reopened.config.name == f'config.resumed-{index:02d}.yaml'


def test_the_config_path_does_not_move_under_you(root):
    experiment = Experiment('demo', root=root)
    experiment.config.write_text('a: 1\n')

    reopened = Experiment.open('demo', root=root)
    before = reopened.config
    reopened.config.write_text('a: 2\n')

    assert reopened.config == before


def test_reopening_a_run_with_no_config_keeps_the_plain_name(root):
    Experiment('demo', root=root)

    assert Experiment.open('demo', root=root).config.name == 'config.yaml'


# ------------------------------------------------------------------ the constructor refuses

def test_the_constructor_refuses_an_existing_directory(root):
    experiment = Experiment('demo', root=root)

    with pytest.raises(FileExistsError):
        clash = Experiment.__new__(Experiment)
        clash.name = 'demo'
        clash.root = root
        clash._run_id = experiment.run_id
        clash.check_existence()


# ------------------------------------------------------------------ ConfigLogger

def test_config_logger_writes_the_params(root):
    experiment = Experiment('demo', root=root)
    params = dict(name='demo', epochs=3, optimizer=dict(learning_rate=1e-3))

    ConfigLogger(experiment.config, params=params)(None)

    assert yaml.safe_load(experiment.config.read_text()) == params


def test_config_logger_keeps_declaration_order(root):
    experiment = Experiment('demo', root=root)
    params = dict(zebra=1, apple=2, mango=3)

    ConfigLogger(experiment.config, params=params)(None)

    assert list(yaml.safe_load(experiment.config.read_text())) == list(params)


def test_config_logger_dumps_an_enum_as_its_value(root):
    from enum import IntEnum, StrEnum

    class Precision(StrEnum):
        FP16 = 'fp16'

    class Workers(IntEnum):
        TWO = 2

    experiment = Experiment('demo', root=root)
    ConfigLogger(
        experiment.config,
        params=dict(precision=Precision.FP16, workers=Workers.TWO),
    )(None)

    assert yaml.safe_load(experiment.config.read_text()) == {
        'precision': 'fp16',
        'workers': 2,
    }


def test_config_logger_dumps_a_torch_device_as_a_string(root):
    import torch

    experiment = Experiment('demo', root=root)
    ConfigLogger(
        experiment.config,
        params=dict(device=torch.device('cuda:1')),
    )(None)

    assert yaml.safe_load(experiment.config.read_text()) == {'device': 'cuda:1'}


def test_config_logger_attached_to_started_writes_once(root):
    experiment = Experiment('demo', root=root)
    engine, epochs = engine_of(3)
    engine.add_event_handler(
        Events.STARTED,
        ConfigLogger(experiment.config, params=dict(epochs=epochs)),
    )
    engine.run([0], max_epochs=epochs)

    assert yaml.safe_load(experiment.config.read_text()) == {'epochs': 3}


# ------------------------------------------------------------------ YamlLogger

def test_yaml_logger_appends_one_document_per_call(root):
    experiment = Experiment('demo', root=root)
    engine, epochs = engine_of(3)
    engine.add_event_handler(
        Events.EPOCH_COMPLETED,
        YamlLogger(
            experiment.metrics,
            output_transform=lambda e: {'epoch': e.state.epoch},
        ),
    )
    engine.run([0], max_epochs=epochs)

    records = list(yaml.safe_load_all(experiment.metrics.read_text()))

    assert records == [{'epoch': 1}, {'epoch': 2}, {'epoch': 3}]


def test_yaml_logger_creates_missing_parents(tmp_path):
    target = tmp_path / 'deep' / 'deeper' / 'metrics.yaml'

    YamlLogger(target, output_transform=lambda e: {})

    assert target.parent.is_dir()


def test_yaml_logger_continues_an_existing_file(root):
    experiment = Experiment('demo', root=root)
    experiment.metrics.write_text('--- {epoch: 1}\n')

    logger = YamlLogger(
        experiment.metrics,
        output_transform=lambda e: {'epoch': 2},
    )
    logger(None)

    records = list(yaml.safe_load_all(experiment.metrics.read_text()))

    assert records == [{'epoch': 1}, {'epoch': 2}]


def test_yaml_logger_keeps_key_order(root):
    experiment = Experiment('demo', root=root)
    record = dict(zebra=1, apple=2, mango=3)

    YamlLogger(experiment.metrics, output_transform=lambda e: record)(None)

    parsed = next(yaml.safe_load_all(experiment.metrics.read_text()))

    assert list(parsed) == list(record)
