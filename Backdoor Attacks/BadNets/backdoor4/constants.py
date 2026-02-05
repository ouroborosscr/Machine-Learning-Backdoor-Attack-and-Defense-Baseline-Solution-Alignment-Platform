alphas = {
    "cifar10": 0.2,
    "gtsrb"  : 0.2,
    "imagenet": 0.4,
    "mnist"  : 0.2
}

dimensions = {
    "cifar10": 3,
    "gtsrb"  : 3,
    "imagenet": 3,
    "mnist"  : 1
}

num_labels = {
    "cifar10": 10,
    "gtsrb"  : 43,
    "imagenet": 20,
    "mnist"  : 10
}

resolution = {
    "cifar10": 32,
    "gtsrb"  : 32,
    "imagenet": 28, # Triggers are upscaled (x8)
    "mnist"  : 28
}

size_limits = {
    "cifar10": 10,
    "gtsrb"  : 10,
    "imagenet": 28,
    "mnist"  : 6
}
