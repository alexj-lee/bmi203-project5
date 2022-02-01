import pytest
import cluster
import numpy as np
import pathlib


@pytest.fixture
def cluster_data():
    data = np.load(pathlib.Path(__file__).resolve().parent / "test_data.npy")
    x = data[:, :-1]
    y = data[:, -1]
    return (x, y)


@pytest.fixture
def mixed_data():
    data = np.load(pathlib.Path(__file__).resolve().parent / "test_data_mixed.npy")
    x = data[:, :-1]
    y = data[:, -1]
    return (x, y)


def test_kmeans_bad_args():

    with pytest.raises(ValueError, match=r"must be a valid pairwise distance metric"):
        kmeans = cluster.KMeans(k=5, metric="notadistance")

    with pytest.raises(TypeError, match=r"k must be numeric"):
        kmeans = cluster.KMeans(k="a")

    with pytest.raises(ValueError, match=r"k must be larger than or equal to 2"):
        kmeans = cluster.KMeans(k=1)

    with pytest.raises(TypeError, match="must be a numeric"):
        kmeans = cluster.KMeans(k=5, max_iter="aa")

    with pytest.raises(ValueError, match="must be positive"):
        kmeans = cluster.KMeans(k=5, tol=-1)


def test_kmeans_output(cluster_data):
    x, y = cluster_data
    k = 3
    kmeans = cluster.KMeans(k=k, max_iter=100, tol=1e-6, metric="euclidean")
    kmeans.fit(x)
    labels_pred = kmeans.predict(x)

    for idx in range(k):
        assert (
            len(np.unique(labels_pred[np.where(y == idx)])) == 1
        ), "Should have found pure clusters for this example."


def test_kmeans_mixed(mixed_data):
    x, y = mixed_data
    k = 3
    kmeans = cluster.KMeans(k=k, max_iter=100, tol=1e-6, metric="euclidean")
    kmeans.fit(x)
    labels_pred = kmeans.predict(x)

    for idx, num in zip(range(k), [2, 2, 1]):
        assert (
            len(np.unique(y[np.where(labels_pred == idx)])) == num
        ), "Should have found the right mixing for this group"

    # for this test we expect label 1 and 2 to be mixed but not label 3; see the test_data_mixed_multipanel.png
