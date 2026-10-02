"""Themed progress columns, one colour per kind of activity.

Each column reads ``task.fields['kind']`` and styles itself from
:data:`TASK_STYLES`, so epoch, training, validation and testing bars stay
visually distinct. :meth:`RichProgressBar.Factory.create_fancy_instance
<stoker.progress.RichProgressBar.Factory.create_fancy_instance>` assembles them
in the intended order; the classes are here for building a different
arrangement.

A task must therefore be created with a ``kind``, and these columns fall back
to the training theme when it is absent. Run this module to see the result:

    python -m stoker.progress.fancy_columns
"""
from rich.progress import (
    Progress,
    ProgressColumn,
    BarColumn,
    Task,
    TextColumn,
)
from rich.text import Text


# ============================================================
# Theme
# ============================================================

TASK_STYLES = {
    'epoch': {
        'icon': '◆',
        'label': 'EPOCH',
        'color': 'magenta',
    },
    'training': {
        'icon': '▶',
        'label': 'TRAIN',
        'color': 'cyan',
    },
    'validation': {
        'icon': '●',
        'label': 'VALID',
        'color': 'green',
    },
    'testing': {
        'icon': '★',
        'label': 'TEST ',
        'color': 'yellow',
    },
}
"""Icon, label and colour per task kind. Keys match ``TaskName``.

Labels are padded to a common width so the bars line up; keep that when
editing. Change this in place to retheme every column at once.
"""


# ============================================================
# Task information
# ============================================================

class FancyColumn(ProgressColumn):
    """The leading column: icon, label, and completed over total."""

    def render(self, task: Task) -> Text:
        config = TASK_STYLES[
            task.fields.get('kind', 'training')
        ]

        text = Text()

        text.append(
            f'{config["icon"]} ',
            style=f'bold {config["color"]}',
        )

        text.append(
            config['label'],
            style=f'bold {config["color"]}',
        )

        text.append(' │ ', style='dim')

        text.append(
            f'{int(task.completed):>4}',
            style='bold',
        )
        text.append(' / ', style='dim')
        text.append(
            f'{int(task.total or 0):<4}',
            style='dim',
        )

        return text


# ============================================================
# Progress bar
# ============================================================

class FancyBarColumn(ProgressColumn):
    """The bar itself, coloured by task kind.

    Holds one :class:`~rich.progress.BarColumn` per kind and delegates to the
    matching one, since a bar's colours are fixed when it is constructed.
    """

    def __init__(self):
        super().__init__()

        self.columns = {
            kind: BarColumn(
                bar_width=24,
                style='grey23',
                complete_style=config['color'],
                finished_style=f'bold {config["color"]}',
            )
            for kind, config in TASK_STYLES.items()
        }

    def render(self, task: Task):
        kind = task.fields.get('kind', 'training')
        return self.columns[kind].render(task)


# ============================================================
# Percentage
# ============================================================

class FancyPercentageColumn(ProgressColumn):
    """The percentage, replaced by a done marker once the task finishes."""

    def render(self, task: Task) -> Text:
        config = TASK_STYLES[
            task.fields.get('kind', 'training')
        ]

        color = config['color']

        if task.finished:
            return Text(
                ' ✓ DONE',
                style=f'bold {color}',
            )

        return Text(
            f' {task.percentage:5.1f}%',
            style=color,
        )


# ============================================================
# Demo
# ============================================================

if __name__ == '__main__':
    progress = Progress(
        FancyColumn(),
        FancyBarColumn(),
        FancyPercentageColumn(),
        TextColumn('[dim italic]{task.fields[postfix]}'),
    )


    with progress:
        epoch = progress.add_task(
            '',
            total=100,
            kind='epoch',
            postfix='best 87.42%',
        )

        training = progress.add_task(
            '',
            total=391,
            kind='training',
            postfix='loss 0.284',
        )

        validation = progress.add_task(
            '',
            total=79,
            kind='validation',
            postfix='top1 84.31%',
        )

        testing = progress.add_task(
            '',
            total=79,
            kind='testing',
            postfix='',
        )

        progress.update(
            epoch,
            completed=42,
        )

        progress.update(
            training,
            completed=287,
        )

        progress.update(
            validation,
            completed=79,
        )

        progress.update(
            testing,
            completed=31,
        )
