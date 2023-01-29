# write your silhouette score unit tests here
import pytest
import cluster
import numpy as np
import pathlib
from sklearn.metrics import silhouette_score as silhouette_score_sklearn


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


def test_silhouette_bad_args(mixed_data):

    # test whether providing a nonexistent distance raises error
    with pytest.raises(ValueError, match=r"must be a valid pairwise distance metric"):
        silhouette = cluster.Silhouette(metric="notadistance")

    def bad_func(x):
        pass

    with pytest.raises(ValueError, match=r"Couldn't call provided metric"):
        silhouette = cluster.Silhouette(metric=bad_func)

    x, y = mixed_data
    silhouette = cluster.Silhouette(
        metric="minkowski"
    )  # test whether class can raise invalid lengths of x, y

    with pytest.raises(ValueError, match=r"X and y must be of same length"):
        silhouette.score(x[:5], y)


def test_silhouette_scoring(mixed_data):
    x, y = mixed_data
    silhouette = cluster.Silhouette(metric="euclidean")
    scores = silhouette.score(x, y)

    # test for correct values with euclidean and cosine distance metric
    mean = scores.mean()
    sum = scores.sum()

    assert np.isclose(mean, 0.06802529015137936), "Silhoette mean is wrong"
    assert np.isclose(sum, 13.60505803027587), "Silhouette sum is wrong"

    scores_sklearn = silhouette_score_sklearn(x, y)
    assert np.isclose(mean, scores_sklearn), "Silhouette doesn't match sklearn"
