import numpy as np
from sklearn.base import BaseEstimator
from qsarcons.hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS
from sklearn.base import is_classifier, is_regressor

def probs_to_class(probs):
    """Convert probability predictions to class labels.

    Handles different shapes:
    - 1D array: threshold 0.5
    - 2D array with 1 column: threshold 0.5
    - 2D array with 2 columns: threshold on second column
    - Multi-class: argmax

    Args:
        probs (np.ndarray): predicted probabilities

    Returns:
        np.ndarray: class labels
    """
    if probs.ndim == 1:
        return (probs > 0.5).astype(int)
    elif probs.shape[1] == 1:
        return (probs[:, 0] > 0.5).astype(int)
    elif probs.shape[1] == 2:
        return (probs[:, 1] > 0.5).astype(int)
    else:
        return np.argmax(probs, axis=1)


class BagWrapper(BaseEstimator):
    """Wrapper for bag-level estimators in Multiple Instance Learning (MIL).

    The estimator is applied on a pooled representation of each bag.
    Supports pooling strategies: mean, max, min, extreme (concatenation of max and min).
    """

    VALID_POOLS = {"mean", "max", "min", "extreme"}

    def __init__(self, estimator, pool="mean"):
        """Initialize BagWrapper.

        Args:
            estimator: sklearn-like estimator with fit() and predict() or predict_proba()
            pool (str or callable): pooling strategy for instances in a bag
        """
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        if not (pool in self.VALID_POOLS or callable(pool)):
            raise ValueError(f"Pooling strategy '{pool}' is not supported.")

        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # determined during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f"{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling"

    def _pooling(self, bags):
        """Pool instances in each bag to a single vector.

        Args:
            bags (list of np.ndarray): list of bags, each bag is [n_instances, n_features]

        Returns:
            np.ndarray: array of pooled bag representations [n_bags, n_features] or [n_bags, 2*n_features] for 'extreme'
        """
        if self.pool == "mean":
            bag_embed = np.asarray([np.mean(bag, axis=0) for bag in bags])
        elif self.pool == "max":
            bag_embed = np.asarray([np.max(bag, axis=0) for bag in bags])
        elif self.pool == "min":
            bag_embed = np.asarray([np.min(bag, axis=0) for bag in bags])
        elif self.pool == "extreme":
            bags_max = np.asarray([np.max(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.min(bag, axis=0) for bag in bags])
            bag_embed = np.concatenate((bags_max, bags_min), axis=1)
        else:
            raise RuntimeError("Unknown pooling strategy.")
        return bag_embed

    def hopt(self, x, y, param_grid, verbose=True):
        """Stepwise hyperparameter optimization."""

        scoring = "roc_auc" if is_classifier(self.estimator) else "r2"

        est_name = self.estimator.__class__.__name__
        param_grid = (
            DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
            if is_classifier(self.estimator)
            else DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
        )

        x = self._pooling(x)
        stepwise_hopt = StepwiseHopt(self.estimator, param_grid, scoring=scoring, verbose=verbose)
        stepwise_hopt.fit(x, y)
        self.estimator = stepwise_hopt.estimator

        return None

    def fit(self, bags, labels):
        """Fit the estimator on bag-level representations.

        Args:
            bags (list of np.ndarray): list of bags
            labels (list or np.ndarray): bag labels

        Returns:
            self
        """
        self.is_classifier = hasattr(self.estimator, "predict_proba")
        bag_embed = self._pooling(bags)
        self.estimator.fit(bag_embed, labels)
        return self

    def predict_proba(self, bags):
        """Predict class probabilities for each bag.

        Args:
            bags (list of np.ndarray): list of bags

        Returns:
            np.ndarray: predicted probabilities
        """
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        bag_embed = self._pooling(bags)
        y_prob = self.estimator.predict_proba(bag_embed)
        return y_prob

    def predict(self, bags):
        """Predict bag labels.

        Args:
            bags (list of np.ndarray): list of bags

        Returns:
            np.ndarray: predicted labels
        """
        if self.is_classifier:
            y_prob = self.predict_proba(bags)
            return probs_to_class(y_prob)
        else:
            bag_embed = self._pooling(bags)
            return self.estimator.predict(bag_embed)

    def get_bag_embedding(self, x):
        """Return pooled bag embeddings.

        Args:
            x (list of np.ndarray): list of bags

        Returns:
            np.ndarray: bag embeddings [n_bags, 1, n_features]
        """
        bag_embed = self._pooling(x)
        return bag_embed[:, None, :]


class InstanceWrapper(BaseEstimator):
    """Wrapper for instance-level estimators in MIL.

    Each instance is assigned the bag label. Bag-level prediction is
    obtained by pooling instance predictions.
    """

    VALID_POOLS = {"mean", "max", "min"}

    def __init__(self, estimator, pool="mean"):
        """Initialize InstanceWrapper.

        Args:
            estimator: sklearn-like estimator with fit() and predict() or predict_proba()
            pool (str or callable): pooling strategy for instance predictions to bag prediction
        """
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # determined during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f"{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling"

    def _pooling(self, inst_pred):
        """Pool instance predictions to obtain bag prediction.

        Args:
            inst_pred (np.ndarray): instance-level predictions

        Returns:
            np.ndarray: bag-level prediction
        """
        inst_pred = np.asarray(inst_pred)

        if callable(self.pool):
            bag_pred = self.pool(inst_pred)
        elif self.pool == "mean":
            bag_pred = np.mean(inst_pred, axis=0)
        elif self.pool == "sum":
            bag_pred = np.sum(inst_pred, axis=0)
        elif self.pool == "max":
            bag_pred = np.max(inst_pred, axis=0)
        elif self.pool == "min":
            bag_pred = np.min(inst_pred, axis=0)
        else:
            raise ValueError(f"Pooling strategy '{self.pool}' is not recognized.")
        return bag_pred

    def hopt(self, x, y, param_grid,  verbose=True):
        """Stepwise hyperparameter optimization."""

        scoring = "roc_auc" if is_classifier(self.estimator) else "r2"

        est_name = self.estimator.__class__.__name__
        param_grid = (
            DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
            if is_classifier(self.estimator)
            else DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
        )

        x_transformed = np.vstack(np.asarray(x, dtype=object)).astype(np.float32)
        y_transformed = np.hstack([np.full(len(bag), lb) for bag, lb in zip(x, y)])

        stepwise_hopt = StepwiseHopt(self.estimator, param_grid, scoring=scoring, verbose=verbose)
        stepwise_hopt.fit(x_transformed, y_transformed)
        self.estimator = stepwise_hopt.estimator

        return None

    def fit(self, bags, labels):
        """Fit estimator on all instances with assigned bag labels.

        Args:
            bags (list of np.ndarray): list of bags
            labels (list or np.ndarray): bag labels

        Returns:
            self
        """
        self.is_classifier = hasattr(self.estimator, "predict_proba")
        bags_transformed = np.vstack(np.asarray(bags, dtype=object)).astype(np.float32)
        labels_transformed = np.hstack([np.full(len(bag), lb) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_transformed, labels_transformed)
        return self

    def predict_proba(self, bags):
        """Predict bag probabilities by pooling instance-level probabilities.

        Args:
            bags (list of np.ndarray): list of bags

        Returns:
            np.ndarray: bag-level probabilities
        """
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        y_pred = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            inst_pred = self.estimator.predict_proba(bag)
            bag_pred = self._pooling(inst_pred)
            y_pred.append(bag_pred)
        return np.array(y_pred)

    def predict(self, bags):
        """Predict bag labels by pooling instance-level predictions.

        Args:
            bags (list of np.ndarray): list of bags

        Returns:
            np.ndarray: predicted labels
        """
        if self.is_classifier:
            y_prob = self.predict_proba(bags)
            return probs_to_class(y_prob)
        else:
            y_pred = []
            for bag in bags:
                bag = bag.reshape(-1, bag.shape[-1])
                inst_pred = self.estimator.predict(bag)
                bag_pred = self._pooling(inst_pred)
                y_pred.append(bag_pred)
            return np.asarray(y_pred)
