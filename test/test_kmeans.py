from re import A
import pytest
import cluster
import numpy as np
import pathlib


@pytest.fixture
def cluster_data():
    data = np.load(pathlib.Path(__file__).resolve().parent / "test_data.npy")
    x = data[:, :-1]  # 1000 data points, 100 features
    y = data[:, -1]
    return (x, y)


@pytest.fixture
def mixed_data():
    data = np.load(pathlib.Path(__file__).resolve().parent / "test_data_mixed.npy")
    x = data[:, :-1]  # 200 data points, 3 features
    y = data[:, -1]
    return (x, y)


def test_kmeans_bad_args():

    # testing normal-ish use cases of what is bad/nonsensical arguments

    with pytest.raises(ValueError, match=r"must be a valid pairwise distance metric"):
        kmeans = cluster.KMeans(k=5, metric="notadistance")

    with pytest.raises(TypeError, match=r"k must be numeric"):
        kmeans = cluster.KMeans(k="a")

    with pytest.raises(ValueError, match=r"k must be larger than or equal to 2"):
        kmeans = cluster.KMeans(k=1)

    with pytest.raises(TypeError, match=r"must be a numeric"):
        kmeans = cluster.KMeans(k=5, max_iter="aa")

    with pytest.raises(ValueError, match=r"must be positive"):
        kmeans = cluster.KMeans(k=5, tol=-1)


def test_kmeans_invalid_fit(cluster_data):
    x, y = cluster_data
    k = 3
    kmeans = cluster.KMeans(k=k, max_iter=100, tol=1e-6, metric="euclidean")

    with pytest.raises(
        RuntimeError,
        match=r"You need to fit the centroids before you get the error value",
    ):
        kmeans.get_error()

    with pytest.raises(ValueError, match=r"can't fit k means with.*data points"):
        kmeans.fit(x[:2])

    with pytest.raises(ValueError, match=r"x must be a 2D matrix"):
        _x = np.ones((50, 50, 50))
        kmeans.fit(_x)

    # test wrong number of features for predicted data
    with pytest.raises(
        ValueError, match=r"must have.*features for input data with current fit."
    ):
        kmeans.fit(x)
        _x = np.c_[x, np.ones(len(x))]  # will be one extra column
        kmeans.predict(_x)


def test_kmeans_output(cluster_data):
    # see test_data_multipanelpng for visualization
    x, y = cluster_data
    k = 3
    kmeans = cluster.KMeans(k=k, max_iter=100, tol=1e-6, metric="euclidean")
    kmeans.fit(x)
    labels_pred = kmeans.predict(x)

    # you should be getting three well separated clusters
    for idx in range(k):
        assert (
            len(np.unique(labels_pred[np.where(y == idx)])) == 1
        ), "Should have found pure clusters for this example."


def test_kmeans_mixed(mixed_data):
    # see test_data_mixed_multipanel.png for visualization; basically they are all squished together
    x, y = mixed_data
    k = 3
    kmeans = cluster.KMeans(k=k, max_iter=100, tol=1e-6, metric="euclidean")
    kmeans.fit(x)
    labels_pred = kmeans.predict(x)

    # should be getting three classes for all three labels lookup (basically all the classes are squished together)
    for idx, num in zip(range(k), (3, 3, 3)):
        assert (
            len(np.unique(labels_pred[np.where(y == idx)])) == num
        ), "Should have found the right mixing for this group"
