import numpy as np
import pytest


def make_bags(n_bags=20, bag_size=6, n_features=4, seed=0, classification=True):
    """Build small synthetic bags and labels for fast, deterministic smoke tests."""
    rng = np.random.RandomState(seed)
    bags = [rng.rand(bag_size, n_features).astype(np.float32) for _ in range(n_bags)]
    if classification:
        labels = rng.randint(0, 2, size=n_bags).astype(np.float32)
    else:
        labels = rng.rand(n_bags).astype(np.float32)
    return bags, labels


@pytest.fixture
def classification_bags():
    """Fixture: small synthetic bags with binary labels."""
    return make_bags(classification=True)


@pytest.fixture
def regression_bags():
    """Fixture: small synthetic bags with continuous labels."""
    return make_bags(classification=False)


FAST_KWARGS = dict(
    max_epochs=3,
    batch_size=8,
    hidden_layer_sizes=(8, 4),
    early_stopping=False,
    accelerator="cpu",
    num_workers=0,
    verbose=False,
)
