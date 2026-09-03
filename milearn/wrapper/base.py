import numpy as np
from sklearn.base import BaseEstimator


def probs_to_class(probs):
    """Convert probabilities to class labels via 0.5 threshold, second-column threshold, or argmax."""
    if probs.ndim == 1:
        return (probs > 0.5).astype(int)
    elif probs.shape[1] == 1:
        return (probs[:, 0] > 0.5).astype(int)
    elif probs.shape[1] == 2:
        return (probs[:, 1] > 0.5).astype(int)
    else:
        return np.argmax(probs, axis=1)


class BagWrapper(BaseEstimator):
    """Wraps a bag-level estimator, applying it to a pooled representation of each bag."""

    VALID_POOLS = {"mean", "max", "min", "extreme"}

    def __init__(self, estimator, pool="mean"):
        """Store the estimator and pooling strategy after validating both."""
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        if not (pool in self.VALID_POOLS or callable(pool)):
            raise ValueError(f"Pooling strategy '{pool}' is not supported.")

        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # determined during fit()

    def __repr__(self):
        """Return a short identifier combining the class, estimator, and pooling names."""
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f"{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling"

    def _pooling(self, bags):
        """Pool each bag's instances into a single vector using the configured strategy."""
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

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Not yet implemented: hyperparameter optimization for wrapped bag-level estimators."""
        raise NotImplementedError("Hyperparameter optimization for wrappers is not implemented yet.")

    def fit(self, bags, labels):
        """Fit the wrapped estimator on pooled bag representations."""
        self.is_classifier = hasattr(self.estimator, "predict_proba")
        bag_embed = self._pooling(bags)
        self.estimator.fit(bag_embed, labels)
        return self

    def predict_proba(self, bags):
        """Predict class probabilities for each bag via the wrapped classifier."""
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        bag_embed = self._pooling(bags)
        y_prob = self.estimator.predict_proba(bag_embed)
        return y_prob

    def predict(self, bags):
        """Predict a label or value for each bag."""
        if self.is_classifier:
            y_prob = self.predict_proba(bags)
            return probs_to_class(y_prob)
        else:
            bag_embed = self._pooling(bags)
            return self.estimator.predict(bag_embed)

    def get_bag_embedding(self, x):
        """Return the pooled bag embeddings, shaped [n_bags, 1, n_features]."""
        bag_embed = self._pooling(x)
        return bag_embed[:, None, :]


class InstanceWrapper(BaseEstimator):
    """Wraps an instance-level estimator, assigning the bag label to every instance in it."""

    VALID_POOLS = {"mean", "max", "min"}

    def __init__(self, estimator, pool="mean"):
        """Store the estimator and pooling strategy used to combine instance predictions."""
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # determined during fit()

    def __repr__(self):
        """Return a short identifier combining the class, estimator, and pooling names."""
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f"{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling"

    def _pooling(self, inst_pred):
        """Pool instance-level predictions into a single bag-level prediction."""
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

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Not yet implemented: hyperparameter optimization for wrapped instance-level estimators."""
        raise NotImplementedError("Hyperparameter optimization for wrappers is not implemented yet.")

    def fit(self, bags, labels):
        """Fit the wrapped estimator on every instance, each assigned its bag's label."""
        self.is_classifier = hasattr(self.estimator, "predict_proba")
        bags_transformed = np.vstack(np.asarray(bags, dtype=object)).astype(np.float32)
        labels_transformed = np.hstack([np.full(len(bag), lb) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_transformed, labels_transformed)
        return self

    def predict_proba(self, bags):
        """Predict bag-level probabilities by pooling instance-level probabilities."""
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
        """Predict a bag-level label or value by pooling instance-level predictions."""
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
