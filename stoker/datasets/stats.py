"""Channel statistics for normalizing common datasets.

Each constant is ``(std, mean)``, in that order, as the names say. Note that
this is the reverse of what :class:`torchvision.transforms.Normalize` takes, so
unpack before use rather than splatting:

    std, mean = torch.tensor(CIFAR100_STD_MEAN)
    transforms.Normalize(mean, std)

Values are per channel, so the MNIST entries are one element tuples and the
rest are three.
"""
MNIST_STD_MEAN = ((0.3081,), (0.1307,))
CIFAR10_STD_MEAN = ((0.2470, 0.2435, 0.2616), (0.4914, 0.4822, 0.4465))
CIFAR100_STD_MEAN = ((0.2675, 0.2565, 0.2761), (0.5071, 0.4867, 0.4408))
IMAGENET_STD_MEAN = ((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
