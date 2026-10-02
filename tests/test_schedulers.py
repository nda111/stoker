"""Shape of the warmup plus cosine schedule."""
import pytest
from ignite.engine import Engine, Events
from torch import nn, optim

from stoker.schedulers import create_cosine_annealing_with_warmup_scheduler

PEAK = 1e-3
END = 1e-8


def trace(iterations, warmup_ratio=0.05, **kwargs):
    """Learning rate after each of `iterations` iterations.

    By default the scheduler is told the length as total_iters; pass any other
    length form as keyword arguments instead.
    """
    optimizer = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.0)
    scheduler = create_cosine_annealing_with_warmup_scheduler(
        optimizer=optimizer,
        start_lr=0.0,
        peak_lr=PEAK,
        end_lr=END,
        warmup_ratio=warmup_ratio,
        **(kwargs or {'total_iters': iterations}),
    )

    history = []

    @Engine
    def engine(engine, batch):
        pass

    engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    engine.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda e: history.append(optimizer.param_groups[0]['lr']),
    )
    engine.run([0] * iterations, max_epochs=1)

    return history


# ------------------------------------------------------------------ shape

@pytest.mark.parametrize('total', [50, 100, 391])
@pytest.mark.parametrize('ratio', [0.0, 0.05, 0.1, 0.2])
def test_the_last_iteration_lands_near_end_lr(total, ratio):
    """Regression guard.

    The cosine has to span one position more than the iterations left to it,
    because the last warmup iteration and the first cosine iteration are the
    same event. Sized without that, the cosine wraps into its next cycle and
    the final iteration gets peak_lr, which is both the largest possible value
    and the one immediately before training ends.
    """
    history = trace(total, ratio)

    assert history[-1] == pytest.approx(END, abs=0.02 * PEAK)
    assert history[-1] < PEAK / 100


@pytest.mark.parametrize('ratio', [0.05, 0.1, 0.2])
def test_warmup_rises_to_the_peak_and_nothing_exceeds_it(ratio):
    total = 100
    history = trace(total, ratio)
    warmup = int(total * ratio)

    assert history[0] < history[warmup - 1]
    assert history[warmup - 1] == pytest.approx(PEAK, rel=1e-6)
    assert max(history) == pytest.approx(PEAK, rel=1e-6)


@pytest.mark.parametrize('ratio', [0.0, 0.05, 0.2])
def test_the_cosine_phase_only_decreases(ratio):
    total = 100
    history = trace(total, ratio)
    tail = history[max(int(total * ratio) - 1, 0):]

    assert all(a >= b for a, b in zip(tail, tail[1:]))


def test_without_warmup_it_starts_at_the_peak():
    history = trace(100, warmup_ratio=0.0)

    assert history[0] == pytest.approx(PEAK, rel=1e-6)


# ------------------------------------------------------------------ length forms

def test_epochs_times_loader_length_matches_total_iters():
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    loader = DataLoader(TensorDataset(torch.zeros(40, 1)), batch_size=4)

    assert len(loader) == 10

    by_loader = trace(50, 0.1, num_epochs=5, data_loader=loader)
    by_total = trace(50, 0.1, total_iters=50)

    assert by_loader == pytest.approx(by_total)


def test_epochs_times_epoch_length_matches_total_iters():
    by_length = trace(50, 0.1, num_epochs=5, epoch_length=10)
    by_total = trace(50, 0.1, total_iters=50)

    assert by_length == pytest.approx(by_total)


def test_a_redundant_length_warns_and_total_iters_wins():
    with pytest.warns(UserWarning, match='num_epochs'):
        by_both = trace(50, 0.1, total_iters=50, num_epochs=999, epoch_length=999)

    assert by_both == pytest.approx(trace(50, 0.1, total_iters=50))


def test_no_length_at_all_is_an_error():
    optimizer = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.0)

    with pytest.raises(ValueError):
        create_cosine_annealing_with_warmup_scheduler(optimizer=optimizer)


def test_epochs_without_a_length_is_an_error():
    optimizer = optim.SGD(nn.Linear(2, 2).parameters(), lr=0.0)

    with pytest.raises(ValueError, match='num_epochs'):
        create_cosine_annealing_with_warmup_scheduler(
            optimizer=optimizer,
            epoch_length=10,
        )
