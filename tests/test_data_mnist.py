import numpy as np

from milearn.data.mnist import create_bags_or, create_bags_and, create_bags_xor, create_bags_reg


def _fake_mnist(n=200, seed=0):
    rng = np.random.RandomState(seed)
    data = rng.rand(n, 8).astype(np.float32)
    targets = rng.randint(0, 10, size=n)
    return data, targets


def test_create_bags_or_shapes_and_labels():
    data, targets = _fake_mnist()
    bags, labels, key_idx = create_bags_or(data, targets, bag_size=5, num_bags=10, key_digit=3, random_state=1)
    assert len(bags) == len(labels) == len(key_idx) == 10
    assert all(bag.shape == (5, 8) for bag in bags)
    assert set(labels).issubset({0, 1})


def test_create_bags_and_positive_bags_contain_both_digits():
    data, targets = _fake_mnist()
    bags, labels, key_idx = create_bags_and(data, targets, bag_size=6, num_bags=10, key_digits=(3, 7), random_state=1)
    assert len(bags) == 10
    for label, idx in zip(labels, key_idx):
        if label == 1:
            assert len(idx) == 2


def test_create_bags_xor_labels_are_binary():
    data, targets = _fake_mnist()
    bags, labels, _ = create_bags_xor(data, targets, bag_size=6, num_bags=10, key_digits=(3, 7), random_state=1)
    assert set(labels).issubset({0, 1})


def test_create_bags_reg_label_is_mean_of_instances():
    data, targets = _fake_mnist()
    bags, labels, digits = create_bags_reg(data, targets, bag_size=4, num_bags=5, bag_agg="mean", random_state=1)
    for label, bag_digits in zip(labels, digits):
        assert np.isclose(label, np.mean(bag_digits))
