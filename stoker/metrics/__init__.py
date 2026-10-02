"""Metrics: reducing within an epoch, and recording across epochs.

:class:`AverageMeter` is an ignite metric that averages over the batches of one
epoch. :class:`MetricState` keeps one value per epoch for each metric and
renders the history as tables.
"""
from .average_meter import AverageMeter
from .metric_state import MetricState
