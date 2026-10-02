"""Averaging whatever the engine returns, over one epoch."""
from typing import Any
import torch, numpy as np
from ignite.metrics import Metric
from ignite.metrics.metric import (
    reinit__is_reduced,
    sync_all_reduce,
)


class AverageMeter(Metric):
    """Running mean of whatever the engine returns, reduced across ranks.

    An ignite :class:`~ignite.metrics.Metric` for the case where the engine
    already produces the quantity to average, such as a training loop
    returning its loss. Attach it as usual:

        meter = AverageMeter()
        meter.attach(trainer, 'loss')

    It averages over every element seen, not over calls, so a per sample loss
    vector contributes its length to the count and the result is the mean over
    samples rather than over batches.

    Note:
        This expects ``output`` to be the value itself, which is why it suits a
        trainer. It does not follow the ``(y_pred, y)`` convention that
        evaluation metrics use; for that, use the ignite metric for the
        quantity, as :class:`~ignite.metrics.Loss` does.
    """

    @reinit__is_reduced
    def reset(self):
        self._sum = 0.0
        self._count = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        """Add one observation, which may hold several values.

        Args:
            output: A tensor or array, whose sum and element count are added;
                a list or tuple, likewise; or a single number.

        Note:
            A tensor is summed to a Python float here, which synchronises with
            the device on every call.
        """
        if torch.is_tensor(output):
            self._sum += output.sum().item()
            self._count += output.numel()
        elif isinstance(output, np.ndarray):
            self._sum += float(np.sum(output))
            self._count += output.size
        elif isinstance(output, (list, tuple)):
            self._sum += sum(output)
            self._count += len(output)
        else:
            self._sum += float(output)
            self._count += 1

    @sync_all_reduce("_sum:SUM", "_count:SUM")
    def compute(self, default: Any = np.nan):
        """The mean over everything seen since :meth:`reset`.

        Sums and counts are totalled across ranks first, so every rank returns
        the same value.

        Args:
            default: What to return when nothing has been seen. Reachable only
                on a direct call, since ignite invokes this with no arguments.

        Returns:
            The mean, or ``default`` on an empty state. Note that ignite
            metrics normally raise
            :exc:`~ignite.exceptions.NotComputableError` in that case; this
            one yields NaN instead.
        """
        if self._count == 0:
            return default

        return self._sum / self._count
