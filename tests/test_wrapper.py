import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from milearn.wrapper import BagWrapper, InstanceWrapper


def test_bag_wrapper_classifier_fit_predict(classification_bags):
    bags, labels = classification_bags
    wrapper = BagWrapper(LogisticRegression(), pool="mean")
    wrapper.fit(bags, labels)
    preds = wrapper.predict(bags)
    assert preds.shape == (len(bags),)
    probs = wrapper.predict_proba(bags)
    assert probs.shape[0] == len(bags)


def test_bag_wrapper_regressor_fit_predict(regression_bags):
    bags, labels = regression_bags
    wrapper = BagWrapper(LinearRegression(), pool="extreme")
    wrapper.fit(bags, labels)
    preds = wrapper.predict(bags)
    assert preds.shape == (len(bags),)


def test_bag_wrapper_rejects_unknown_pool():
    with pytest.raises(ValueError):
        BagWrapper(LinearRegression(), pool="bogus")


def test_bag_wrapper_hopt_is_not_implemented(classification_bags):
    bags, labels = classification_bags
    wrapper = BagWrapper(LogisticRegression(), pool="mean")
    with pytest.raises(NotImplementedError):
        wrapper.hopt(bags, labels)


def test_instance_wrapper_classifier_fit_predict(classification_bags):
    bags, labels = classification_bags
    wrapper = InstanceWrapper(LogisticRegression(), pool="mean")
    wrapper.fit(bags, labels)
    preds = wrapper.predict(bags)
    assert preds.shape == (len(bags),)


def test_instance_wrapper_hopt_is_not_implemented(classification_bags):
    bags, labels = classification_bags
    wrapper = InstanceWrapper(LogisticRegression(), pool="mean")
    with pytest.raises(NotImplementedError):
        wrapper.hopt(bags, labels)
