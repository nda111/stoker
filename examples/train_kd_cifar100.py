import os

import torch
from torch import nn, optim
from torchvision.datasets import CIFAR100
from torchvision import transforms

import timm
from timm.models.resnet import ResNet

from ignite import utils as iutils
from ignite import distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, TopKCategoricalAccuracy
from ignite.handlers import Checkpoint, DiskSaver

from stoker.config import argconfig, ConfigMixin, Annotated, Argument, default
from stoker.utils import graceful_run
from stoker.datasets.stats import CIFAR100_STD_MEAN
from stoker.metrics import MetricState, AverageMeter
from stoker.schedulers import create_cosine_annealing_with_warmup_scheduler
from stoker.progress import RichProgressBar
from stoker.tracking import Experiment, ConfigLogger, YamlLogger


# ========================================================
# Arguments
# ========================================================

filename = os.path.splitext(os.path.split(__file__)[1])[0]

@argconfig(
    prog=f'python -m {filename}',
    description='Train',
)
class Config(ConfigMixin):
    name: Annotated[
        str,
        Argument('name'),
    ]

    device: torch.device = torch.device('cuda:0')
    seed: int | None = None

    resume: Annotated[
        str | None,
        Argument(
            '--resume',
            metavar='RUN_ID',
            nargs='?',
            const='latest',
        ),
    ] = None

    epochs: int = 100
    batch_size: int = 512

    student: str = 'resnet18'
    teacher: str = 'resnet50.tv_in1k'

    use_amp: Annotated[
        bool,
        Argument(
            '--use-amp',
            '-amp',
            action='store_true',
        ),
    ] = False

    @argconfig
    class OptimizerConfig(ConfigMixin):
        learning_rate: float = 1e-3
        weight_decay: float = 1e-2

    optimizer: OptimizerConfig = default()

    @argconfig
    class SchedulerConfig(ConfigMixin):
        warmup_ratio: float = 0.05
        end_ratio: float = 1e-5

    scheduler: SchedulerConfig = default()

    @argconfig
    class DistillationConfig(ConfigMixin):
        layers: Annotated[
            list[int],
            Argument(
                '--layers',
                type=int,
                nargs='*',
            )
        ] = None  # None for no layers, [-1] for all layers.
        logits: Annotated[
            bool,
            Argument(
                '--logits',
                action='store_true',
            )
        ] = False
        weight: float = 0.05
        temperature: float = 0.5

    distill: DistillationConfig = default()

config = Config.parse_args()
config.dump()

if config.resume:
    experiment = Experiment.open(config.name, run_id=config.resume)
else:
    experiment = Experiment(name=config.name)

if isinstance(config.seed, int):
    iutils.manual_seed(config.seed)

# ========================================================
# Data
# ========================================================

dataset_kwargs = dict(
    root='data',
)
std, mean = CIFAR100_STD_MEAN
common_transform = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
]
trainset = CIFAR100(
    **dataset_kwargs,
    train=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] + common_transform),
)
validset = CIFAR100(
    **dataset_kwargs,
    train=False,
    transform=transforms.Compose(common_transform),
)

loader_kwargs = dict(
    batch_size=config.batch_size,
    pin_memory=True,
)
train_loader = idist.auto_dataloader(
    trainset, 
    **loader_kwargs,
    shuffle=True,
)
valid_loader = idist.auto_dataloader(
    validset, 
    **loader_kwargs,
    shuffle=False,
)

# ========================================================
# Model & Optimizer
# ========================================================

student = timm.create_model(
    config.student,
    pretrained=False,
    num_classes=100,
)
if isinstance(student, ResNet):
    student.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    student.maxpool = nn.Identity()
student = idist.auto_model(student)
teacher = timm.create_model(
    config.teacher,
    pretrained=False,
    num_classes=100,
)
teacher = idist.auto_model(teacher).eval().requires_grad_(False)

optimizer = optim.AdamW(
    student.parameters(),
    lr=0,
    weight_decay=config.optimizer.weight_decay,
)
optimizer = idist.auto_optim(optimizer)
scheduler = create_cosine_annealing_with_warmup_scheduler(
    optimizer=optimizer,
    start_lr=0,
    peak_lr=config.optimizer.learning_rate,
    end_lr=config.optimizer.learning_rate * config.scheduler.end_ratio,
    num_epochs=config.epochs,
    data_loader=train_loader,
    warmup_ratio=config.scheduler.warmup_ratio,
)

scaler = torch.amp.GradScaler(config.device.type, enabled=config.use_amp)

# ========================================================
# Metrics & UI
# ========================================================

progress_bar = RichProgressBar.Factory.create_fancy_instance()

train_loss_ce_meter = AverageMeter(output_transform=lambda y: y['ce'])
train_loss_kd_meter = AverageMeter(output_transform=lambda y: y['kd'])
valid_loss_ce_meter = AverageMeter(output_transform=lambda y: y[2]['ce'])
valid_loss_kd_meter = AverageMeter(output_transform=lambda y: y[2]['kd'])
top1_meter = Accuracy(output_transform=lambda y: y[:2])
top5_meter = TopKCategoricalAccuracy(k=5, output_transform=lambda y: y[:2])

metric_state = MetricState(
    training=dict(
        lr=[],
        loss_ce=[],
        loss_kd=[],
    ),
    validation=dict(
        loss_ce=[],
        loss_kd=[],
        top1=[],
        top5=[],
    )
)

# ========================================================
# Loops
# ========================================================

@Engine
def trainer(engine: Engine, batch: tuple[torch.Tensor, ...]):
    student.train()
    T = config.distill.temperature
    weight_kd = config.distill.weight
    weight_ce = 1 - weight_kd

    x, y = batch
    x = x.to(config.device, non_blocking=True)
    y = y.to(config.device, non_blocking=True)  # (B, C) 

    with torch.autocast(device_type=config.device.type, enabled=config.use_amp):
        with torch.no_grad():
            y_teacher: torch.Tensor = teacher(x)
            sm_teacher = (y_teacher / T).log_softmax(dim=-1)
        y_student: torch.Tensor = student(x)
        sm_student = (y_student / T).log_softmax(dim=-1)
        
        loss_ce_vector = nn.functional.cross_entropy(
            y_student, y, reduction='none',
        )  # (B,)
        loss_kd_vector = nn.functional.kl_div(
            sm_student, sm_teacher, reduction='none', log_target=True,
        ).sum(dim=-1) * (T ** 2)  # (B,)
        loss_ce = loss_ce_vector.mean()
        loss_kd = loss_kd_vector.mean()
        loss = loss_ce * weight_ce + loss_kd * weight_kd

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    progress_bar.set_postfix({
        'loss_ce': f'{loss_ce:.4f}',
        'loss_kd': f'{loss_kd:.4f}',
    }, where='training')

    return {
        'ce': loss_ce_vector.detach(),
        'kd': loss_kd_vector.detach(),
    }

@Engine
@torch.no_grad()
def evaluator(engine: Engine, batch: tuple[torch.Tensor, ...]):
    student.eval()
    T = config.distill.temperature

    x, y = batch
    x = x.to(config.device, non_blocking=True)
    y = y.to(config.device, non_blocking=True)

    with torch.autocast(device_type=config.device.type, enabled=config.use_amp):
        y_student: torch.Tensor = student(x)
        y_teacher: torch.Tensor = teacher(x)
        sm_student = (y_student / T).log_softmax(dim=-1)
        sm_teacher = (y_teacher / T).log_softmax(dim=-1)
        
        loss_ce_vector = nn.functional.cross_entropy(
            y_student, y, reduction='none',
        )  # (B,)
        loss_kd_vector = nn.functional.kl_div(
            sm_student, sm_teacher, reduction='none', log_target=True,
        ).sum(dim=-1) * (T ** 2)  # (B,)

    return y_student, y, {
        'ce': loss_ce_vector,
        'kd': loss_kd_vector,
    }

# ========================================================
# Logging & Checkpointing
# ========================================================

# Loggers
config_logger = ConfigLogger(
    experiment.config,
    params=config.to_dict(),
)

yaml_logger = YamlLogger(
    experiment.metrics,
    output_transform=lambda engine: {
        'epoch': engine.state.epoch,
        'training': {
            'lr': optimizer.param_groups[0]['lr'],
            **engine.state.metrics,
        },
        'validation': evaluator.state.metrics,
    },
)

# Checkpoints
ckpt_components = dict(
    model=student,
    optimizer=optimizer,
    scheduler=scheduler,
    trainer=trainer,
    scaler=scaler,
    metric=metric_state,
)

periodic_ckpt = Checkpoint(
    to_save=ckpt_components,
    save_handler=DiskSaver(
        dirname=experiment.periodic_checkpoints,
        create_dir=True,
    ),
    filename_pattern='epoch_{global_step:03d}',
    global_step_transform=lambda *_: trainer.state.epoch,
    n_saved=None,
)

latest_ckpt = Checkpoint(
        to_save=ckpt_components,
    save_handler=DiskSaver(
        dirname=experiment.latest_checkpoints,
        create_dir=True,
    ),
    filename_pattern='last',
    n_saved=1,
)

best_ckpt = Checkpoint(
    to_save=ckpt_components,
    save_handler=DiskSaver(
        dirname=experiment.best_checkpoints,
        create_dir=True,
    ),
    filename_pattern='epoch_{global_step:03d}_{score}',
    global_step_transform=lambda *_: trainer.state.epoch,
    score_function=lambda engine: engine.state.metrics['top1'],
    score_name='val_top1',
    n_saved=3,
)

# Resume
if experiment.resumed:
    checkpoint = experiment.latest_checkpoints / 'last'

    if checkpoint.is_file():
        Checkpoint.load_objects(
            to_load=ckpt_components,
            checkpoint=torch.load(
                checkpoint,
                map_location=config.device,
                weights_only=False,
            ),
        )
        progress_bar.print(
            f'Resumed {experiment.run_id} from epoch {trainer.state.epoch}.'
        )

    else:
        progress_bar.print(
            f'No checkpoint in {experiment.latest_checkpoints}, '
            f'quitting training.'
        )
        exit()

# ========================================================
# Attachments
# ========================================================

# Metrics
train_loss_ce_meter.attach(trainer, 'loss_ce')
train_loss_kd_meter.attach(trainer, 'loss_kd')
valid_loss_ce_meter.attach(evaluator, 'loss_ce')
valid_loss_kd_meter.attach(evaluator, 'loss_kd')
top1_meter.attach(evaluator, 'top1')
top5_meter.attach(evaluator, 'top5')

# Progress bar
progress_bar.attach(
    training=trainer,
    validation=evaluator,
)

# Training flow
trainer.add_event_handler(
    Events.STARTED,
    config_logger,
)

trainer.add_event_handler(
    Events.ITERATION_COMPLETED,
    scheduler,
)

@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(engine: Engine):
    evaluator.run(valid_loader)

@trainer.on(Events.EPOCH_COMPLETED)
def update_training_metrics(engine: Engine):
    metric_state.update(
        epoch=engine.state.epoch,
        training=dict(
            lr=optimizer.param_groups[0]['lr'],
            **engine.state.metrics,
        ),
        validation=evaluator.state.metrics,
    )
    history, _ = metric_state.summary(best_key='valid.top1;big')
    progress_bar.print(history)

trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    yaml_logger,
)

# Checkpointing
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    latest_ckpt,
)
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(every=10),
    periodic_ckpt,
)
evaluator.add_event_handler(
    Events.COMPLETED,
    best_ckpt,
)

# ========================================================
# Execution
# ========================================================

graceful_run(
    trainer,
    data=train_loader,
    max_epochs=config.epochs,
    handlers={
        KeyboardInterrupt: 'Interrupted by user. Terminating training...',
    },
)
