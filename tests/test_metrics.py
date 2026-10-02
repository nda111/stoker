"""Metric history, the summary tables, and serialization."""
import pytest
import torch

from stoker.metrics import AverageMeter, MetricState
from stoker.metrics.metric_state import MetricGroup


def filled(epochs=3):
    state = MetricState(
        training=dict(loss=[], lr=0.0),
        validation=dict(top1=[]),
    )

    for epoch in range(1, epochs + 1):
        state.update(
            epoch=epoch,
            training=dict(loss=1.0 / epoch, lr=1e-3),
            validation=dict(top1=epoch / 10),
        )

    return state


# ------------------------------------------------------------------ MetricGroup

def test_group_allows_attribute_access():
    group = MetricGroup(dict(loss=0.5))

    assert group.loss == 0.5

    group.loss = 0.25

    assert group['loss'] == 0.25


def test_group_rejects_unsupported_values():
    group = MetricGroup()

    with pytest.raises(TypeError):
        group['model'] = object()


def test_group_rejects_non_string_keys():
    group = MetricGroup()

    with pytest.raises(TypeError):
        group[1] = 0.5


def test_group_reports_a_missing_attribute_as_attribute_error():
    with pytest.raises(AttributeError):
        MetricGroup().nothing_here


# ------------------------------------------------------------------ schema

def test_a_list_is_historical_and_a_scalar_is_not():
    state = filled()

    assert state.training['loss'] == [1.0, 0.5, pytest.approx(1 / 3)]
    assert state.training['lr'] == 1e-3


def test_history_stays_as_long_as_the_epoch_list():
    state = filled(4)

    assert len(state.epoch) == 4
    assert len(state.training['loss']) == 4
    assert len(state.validation['top1']) == 4


def test_a_missing_historical_metric_is_padded():
    state = MetricState(training=dict(loss=[], extra=[]))
    state.update(epoch=1, training=dict(loss=0.5))

    assert state.training['extra'] == [None]


def test_a_metric_outside_the_schema_is_rejected():
    state = MetricState(training=dict(loss=[]))

    with pytest.raises(KeyError, match='train.nope'):
        state.update(epoch=1, training=dict(nope=1.0))


def test_nothing_is_committed_when_validation_fails():
    state = MetricState(training=dict(loss=[]), testing=dict(top1=[]))

    with pytest.raises(KeyError):
        state.update(epoch=1, training=dict(loss=0.5), testing=dict(nope=1))

    assert state.epoch == []
    assert state.training['loss'] == []


def test_strict_requires_every_metric():
    state = MetricState(training=dict(loss=[], lr=[]))

    with pytest.raises(KeyError, match='train.lr'):
        state.update(epoch=1, training=dict(loss=0.5), strict=True)


# ------------------------------------------------------------------ epochs

def test_epochs_must_increase():
    state = filled(2)

    for epoch in [2, 1]:
        with pytest.raises(ValueError, match='increasing'):
            state.update(epoch=epoch, training=dict(loss=0.1))


def test_epoch_must_be_an_int():
    state = MetricState(training=dict(loss=[]))

    for epoch in [1.0, '1', torch.tensor(1)]:
        with pytest.raises(TypeError):
            state.update(epoch=epoch, training=dict(loss=0.1))


# ------------------------------------------------------------------ summary

def test_summary_returns_two_tables():
    history, current = filled().summary(best_key='valid.top1;big')
    headers = [str(column.header) for column in history.columns]

    assert headers[0] == 'epoch'
    assert 'train.loss' in headers
    assert any('valid.top1' in header for header in headers)
    assert history.row_count > 0
    assert current.row_count == 1          # lr, the only non historical metric


def test_recent_limits_the_rows_shown():
    state = filled(10)
    history, _ = state.summary(recent=2, best=1, best_key='valid.top1;big')

    assert history.row_count == 3          # two recent, one best


def test_best_zero_omits_the_section():
    state = filled(10)
    history, _ = state.summary(recent=2, best=0, best_key='valid.top1;big')

    assert history.row_count == 2


def test_excluded_keys_drop_columns():
    history, _ = filled().summary(
        best_key='valid.top1;big',
        excluded_keys=['epoch', 'train.loss'],
    )

    headers = [column.header for column in history.columns]

    assert all('epoch' not in str(h) for h in headers)
    assert all('train.loss' not in str(h) for h in headers)


def test_state_table_row_order_follows_declaration():
    state = MetricState(training=dict(lr=0.0, wd=0.0, momentum=0.0, eps=0.0))
    state.update(epoch=1, training=dict(lr=1.0))

    _, current = state.summary(best=0)
    names = list(current.columns[0]._cells)

    assert names == ['train.lr', 'train.wd', 'train.momentum', 'train.eps']


@pytest.mark.parametrize(
    'best_key',
    [
        'training.loss;small',     # full group name, not the short prefix
        'train.loss;epoch',        # epoch mode only pairs with epoch
        'epoch;small',
        'train.loss',              # no mode at all
        'train.lr;small',          # not a historical metric
        'train.nope;small',
    ],
)
def test_a_malformed_best_key_is_rejected(best_key):
    with pytest.raises((KeyError, ValueError)):
        filled().summary(best_key=best_key)


def test_epoch_can_be_the_ranking_key():
    history, _ = filled().summary(best_key='epoch;epoch')

    assert history.row_count > 0


def test_negative_limits_are_rejected():
    for kwargs in [dict(recent=-1), dict(best=-1)]:
        with pytest.raises(ValueError):
            filled().summary(**kwargs)


# ------------------------------------------------------------------ serialization

def test_state_dict_round_trip():
    state = filled(3)
    exported = state.state_dict()

    restored = MetricState(
        training=dict(loss=[], lr=0.0),
        validation=dict(top1=[]),
    )
    restored.load_state_dict(exported)

    assert restored.epoch == state.epoch
    assert restored.training['loss'] == state.training['loss']
    assert restored.validation['top1'] == state.validation['top1']


def test_state_dict_is_a_snapshot():
    state = filled(2)
    exported = state.state_dict()

    state.update(epoch=3, training=dict(loss=0.1), validation=dict(top1=0.9))

    assert exported['epoch'] == [1, 2]
    assert exported['training']['loss'] == [1.0, 0.5]


def test_a_state_that_declared_nothing_exports_nothing():
    assert MetricState().state_dict() == {}


def test_a_fresh_state_exports_its_declared_metrics_empty():
    exported = MetricState(training=dict(loss=[])).state_dict()

    assert exported == {'training': {'loss': []}}
    assert 'epoch' not in exported


def test_load_state_dict_rejects_unexpected_keys():
    state = MetricState(training=dict(loss=[]))

    with pytest.raises(KeyError, match='nope'):
        state.load_state_dict({'nope': []})


def test_load_state_dict_rejects_a_non_mapping():
    state = MetricState(training=dict(loss=[]))

    with pytest.raises(TypeError):
        state.load_state_dict([1, 2])


def test_resuming_keeps_the_epoch_constraint():
    state = filled(3)
    exported = state.state_dict()

    restored = MetricState(
        training=dict(loss=[], lr=0.0),
        validation=dict(top1=[]),
    )
    restored.load_state_dict(exported)
    restored.update(epoch=4, training=dict(loss=0.2), validation=dict(top1=0.4))

    assert restored.epoch == [1, 2, 3, 4]


# ------------------------------------------------------------------ AverageMeter

def test_average_meter_averages_over_elements_not_calls():
    meter = AverageMeter()
    meter.reset()
    meter.update(torch.tensor([1.0, 3.0]))
    meter.update(torch.tensor([5.0]))

    assert meter.compute() == pytest.approx(3.0)


@pytest.mark.parametrize(
    'output, expected',
    [
        (torch.tensor([2.0, 4.0]), 3.0),
        ([1.0, 2.0, 3.0], 2.0),
        ((4.0, 6.0), 5.0),
        (7.0, 7.0),
    ],
)
def test_average_meter_accepts_several_shapes(output, expected):
    meter = AverageMeter()
    meter.reset()
    meter.update(output)

    assert meter.compute() == pytest.approx(expected)


def test_average_meter_returns_the_default_when_empty():
    meter = AverageMeter()
    meter.reset()

    assert meter.compute(default=-1) == -1


def test_average_meter_reset_clears():
    meter = AverageMeter()
    meter.reset()
    meter.update(10.0)
    meter.reset()

    assert meter.compute(default=None) is None
