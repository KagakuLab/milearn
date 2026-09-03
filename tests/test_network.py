import copy

import numpy as np
import pytest

from milearn.network.classifier import BagNetworkClassifier, InstanceNetworkClassifier
from milearn.network.regressor import BagNetworkRegressor
from milearn.network.classifier import BagWrapperMLPNetworkClassifier

from conftest import FAST_KWARGS


def test_bag_network_classifier_fit_predict(classification_bags):
    bags, labels = classification_bags
    model = BagNetworkClassifier(pool="mean", **FAST_KWARGS)
    model.fit(bags, labels)
    preds = model.predict(bags)
    assert preds.shape == (len(bags),)
    assert np.all((preds >= 0) & (preds <= 1))


def test_bag_network_regressor_fit_predict(regression_bags):
    bags, labels = regression_bags
    model = BagNetworkRegressor(pool="max", **FAST_KWARGS)
    model.fit(bags, labels)
    preds = model.predict(bags)
    assert preds.shape == (len(bags),)


def test_instance_network_invalid_pool_raises_clear_error(classification_bags):
    bags, labels = classification_bags
    model = InstanceNetworkClassifier(pool="bogus", **FAST_KWARGS)
    with pytest.raises(TypeError):
        model.fit(bags, labels)


def test_bag_wrapper_mlp_fit_predict_consistent_for_non_mean_pool(classification_bags):
    bags, labels = classification_bags
    model = BagWrapperMLPNetworkClassifier(pool="max", **FAST_KWARGS)
    model.fit(bags, labels)
    preds = model.predict(bags)
    assert preds.shape == (len(bags),)


def test_hopt_does_not_mutate_caller_param_grid(classification_bags):
    bags, labels = classification_bags
    model = BagNetworkClassifier(**FAST_KWARGS)

    tiny_grid = {**FAST_KWARGS, "pool": ["mean", "max"]}
    tiny_grid_before = copy.deepcopy(tiny_grid)

    model.hopt(bags, labels, param_grid=tiny_grid, verbose=False)

    assert tiny_grid == tiny_grid_before
