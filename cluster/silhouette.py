import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance


class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        self.metric = metric  # we want to store it as a string because the string referenced versions of the fns are in C

        if metric is None or isinstance(metric, str):
            metric = distance.__dict__.get(metric, None)
            if metric is None:
                raise ValueError(
                    'Argument "metric" must be a valid pairwise distance metric found in scipy.spatial.distance.'
                )

            if callable(metric) is False:
                raise ValueError(
                    f"Metric must be a function found in scipy.spatial.distance. Found object is of type {type(metric)}"
                )

        if callable(metric):
            try:
                metric([0, 1], [0, 1])
            except:
                raise ValueError("Couldn't call provided metric function with two args")

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if len(X) != len(y):
            raise ValueError("X and y must be of same length.")

        # TODO: lookup if number of pts in a cluster is 0, for which s(i) = 0 according to wikipedia
        pdist = cdist(X, X, metric=self.metric)
        assert np.allclose(pdist, pdist.T)

        clusters = np.unique(y)
        scores = np.zeros(len(X))
        indices = np.arange(len(X))

        for idx in range(len(X)):
            cluster_identity = y[idx]

            lookup = np.intersect1d(
                np.where(y == cluster_identity), np.where(indices != idx)
            )
            a = pdist[idx, lookup].mean()

            cluster_b_dict = {}
            for clust_id in clusters:
                if clust_id == cluster_identity:
                    continue

                lookup = np.where(y == clust_id)
                _b = pdist[idx, lookup].mean()
                cluster_b_dict[clust_id] = _b

            b = cluster_b_dict[min(cluster_b_dict, key=cluster_b_dict.get)]

            scores[idx] = (b - a) / max(a, b)
        return scores
