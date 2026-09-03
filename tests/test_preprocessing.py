import numpy as np

from milearn.preprocessing import BagMinMaxScaler, BagStandardScaler


def test_bag_minmax_scaler_ranges():
    bags = [np.array([[0.0, 10.0], [2.0, 20.0]]), np.array([[1.0, 15.0]])]
    scaler = BagMinMaxScaler()
    scaled = scaler.fit_transform(bags)
    stacked = np.vstack(scaled)
    assert stacked.min() >= 0.0
    assert stacked.max() <= 1.0


def test_bag_standard_scaler_shapes_preserved():
    bags = [np.random.rand(3, 4), np.random.rand(5, 4)]
    scaler = BagStandardScaler()
    scaled = scaler.fit_transform(bags)
    assert [b.shape for b in scaled] == [b.shape for b in bags]
