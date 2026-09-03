import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from numpy import ndarray
from typing import Tuple
from typing import List


def load_mnist(flatten: bool = True) -> Tuple[ndarray, ndarray]:
    """Download MNIST and return images (optionally flattened) alongside their digit labels."""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data = mnist.data.numpy()
    targets = mnist.targets.numpy()

    if flatten:
        data = data.reshape((data.shape[0], -1))
    return data, targets


def create_bags_or(data: ndarray, targets: ndarray, bag_size: int = 10, num_bags: int = 1000, key_digit: int = 3, key_instances_per_bag: int = 1, random_state: int = 42) -> Tuple[List[ndarray], List[int], List[List[int]]]:
    """Build bags labeled positive if they contain at least one instance of the key digit."""
    rng = np.random.RandomState(random_state)
    key_indices_all = np.where(targets == key_digit)[0]
    non_key_indices_all = np.where(targets != key_digit)[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            key_sample_indices = rng.choice(key_indices_all, size=key_instances_per_bag, replace=False)
            remaining = bag_size - key_instances_per_bag
            non_key_sample_indices = rng.choice(non_key_indices_all, size=remaining, replace=False)
            full_indices = np.concatenate([key_sample_indices, non_key_sample_indices])
            rng.shuffle(full_indices)
            key_pos_in_bag = [i for i, idx in enumerate(full_indices) if idx in key_sample_indices]
            label = 1
        else:
            full_indices = rng.choice(non_key_indices_all, size=bag_size, replace=False)
            key_pos_in_bag = []
            label = 0

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos_in_bag)

    return bags, bag_labels, key_indices_per_bag


def create_bags_and(data, targets, bag_size=10, num_bags=1000, key_digits=(3, 7), random_state=42):
    """Build bags labeled positive only if they contain both key digits."""
    rng = np.random.RandomState(random_state)
    idx_key1 = np.where(targets == key_digits[0])[0]
    idx_key2 = np.where(targets == key_digits[1])[0]
    idx_nonkey = np.where(~np.isin(targets, key_digits))[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            idx1 = rng.choice(idx_key1)
            idx2 = rng.choice(idx_key2)
            remaining = bag_size - 2
            idx_other = rng.choice(idx_nonkey, size=remaining, replace=False)
            full_indices = np.array([idx1, idx2] + list(idx_other))
            rng.shuffle(full_indices)
            label = 1
            key_pos = [i for i, idx in enumerate(full_indices) if idx in [idx1, idx2]]
        else:
            while True:
                full_indices = rng.choice(len(targets), size=bag_size, replace=False)
                bag_targets = targets[full_indices]
                if not (key_digits[0] in bag_targets and key_digits[1] in bag_targets):
                    break
            label = 0
            key_pos = []

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos)

    return bags, bag_labels, key_indices_per_bag


def create_bags_xor(data, targets, bag_size=10, num_bags=1000, key_digits=(3, 7), random_state=42):
    """Build bags labeled positive if they contain exactly one of the two key digits."""
    rng = np.random.RandomState(random_state)
    idx_key1 = np.where(targets == key_digits[0])[0]
    idx_key2 = np.where(targets == key_digits[1])[0]
    idx_nonkey = np.where(~np.isin(targets, key_digits))[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            use_digit = key_digits[rng.randint(2)]
            idx_key = rng.choice(idx_key1 if use_digit == key_digits[0] else idx_key2)
            idx_other = rng.choice(idx_nonkey, size=bag_size - 1, replace=False)
            full_indices = np.array([idx_key] + list(idx_other))
            rng.shuffle(full_indices)
            label = 1
            key_pos = [i for i, idx in enumerate(full_indices) if targets[idx] == use_digit]
        else:
            while True:
                full_indices = rng.choice(len(targets), size=bag_size, replace=False)
                bag_targets = targets[full_indices]
                count_1 = np.sum(bag_targets == key_digits[0])
                count_2 = np.sum(bag_targets == key_digits[1])
                if (count_1 > 0 and count_2 > 0) or (count_1 == 0 and count_2 == 0):
                    break
            label = 0
            key_pos = []

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos)

    return bags, bag_labels, key_indices_per_bag


def create_bags_reg(data: ndarray, targets: ndarray, bag_size: int = 5, num_bags: int = 1000, bag_agg: str = "mean", random_state: int = 42) -> Tuple[List[ndarray], List[float], List[List[int]]]:
    """Build regression bags whose label is the mean or sum of their instances' target values."""
    if bag_agg == "mean":
        agg_func = np.mean
    elif bag_agg == "sum":
        agg_func = np.sum
    else:
        raise TypeError(f"Unknown value for bag_agg: {bag_agg}")

    rng = np.random.RandomState(random_state)
    indices = np.arange(len(data))
    bags, labels, instance_digits = [], [], []

    for _ in range(num_bags):
        selected_indices = rng.choice(indices, size=bag_size, replace=False)
        bag = data[selected_indices]
        digits = targets[selected_indices]
        label = agg_func(digits)
        bags.append(bag)
        labels.append(label.item())
        instance_digits.append(digits.tolist())

    return bags, labels, instance_digits


def show_digit(vector, title=None):
    """Display a flattened 28x28 MNIST digit as an image, with an optional title."""
    if vector.shape[0] != 784:
        raise ValueError("Expected a vector of length 784.")

    image = vector.reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def visualize_bag_with_weights(bag, weights, digits=None, title=None, cmap='gray', sort=False):
    """Display every image in a bag as a grid, annotated with its attention weight."""
    bag = np.array(bag)
    weights = np.array(weights)

    if sort and digits is not None:
        digits = np.array(digits)
        sort_idx = np.argsort(digits)
        bag = bag[sort_idx]
        weights = weights[sort_idx]

    bag_size = len(bag)
    cols = min(5, bag_size)
    rows = (bag_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten()

    for i in range(bag_size):
        image = bag[i].reshape(28, 28)
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(f"{weights[i]:.2f}", fontsize=12)
        axes[i].axis("off")

    for i in range(bag_size, len(axes)):
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
