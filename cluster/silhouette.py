import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance


class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                The name of the distance metric that we want to do. We are going to explicitly
        """

        self.metric = metric
        # we want to store it as a string because the string referenced versions of the fns are written in C for scipy.distances; for whatever reason using the non-string indexed version uses the python/numpy implementation

        if metric is None or isinstance(metric, str):
            metric = distance.__dict__.get(metric, None)
            if metric is None:
                raise ValueError(
                    'Argument "metric" (str) must be a valid pairwise distance metric found in scipy.spatial.distance.'
                )

            if callable(metric) is False:
                raise ValueError(
                    f"Metric must be a function found in scipy.spatial.distance. Found object is of type {type(metric)}"
                )

        elif callable(metric):
            try:
                x = np.array([1, 2])[None, :]  # will be of shape 1, 2
                y = np.array([1, 4])[None, :]  # will be of shape 1, 2
                metric(x, y)
                self.metric = metric  # if its a callable let's just assume that the user provided a reasonable metric function
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

        pdist = cdist(X, X, metric=self.metric)  # pairwise distance matrix

        assert np.allclose(pdist, pdist.T), "Pairwise distance matrix was asymmetric."

        clusters = np.unique(y)  # list of clusters

        scores = np.zeros(
            len(X)
        )  # empty array to hold the silhouette per-data-pt scores

        if len(clusters) == 1:  # if theres only one cluster, we'll just return all ones
            scores.fill(1)
            return scores

        indices = np.arange(
            len(X)
        )  # indices of the data points, so we can figure out and array index which data points are part of cluster i

        for idx in range(len(X)):
            cluster_identity = y[
                idx
            ]  # get the actual cluster y is in, so we can use this to indx the rows that are same cluster and not same cluster

            # once we have cluster_identity, we look for the indices where y is the right cluster and exclude the diagonal entry (where i=j)
            lookup = np.intersect1d(
                np.where(y == cluster_identity), np.where(indices != idx)
            )

            a = pdist[
                idx, lookup
            ].mean()  # this will be the distances for the idx-th data point where the class is the same, except the diagonal

            # here we loop over the other clusters (the ones that are not the same as the idx-th data pt)
            cluster_b_dict = {}
            for clust_id in clusters:
                if clust_id == cluster_identity:
                    continue

                lookup = np.where(y == clust_id)
                # get the distances from idxth data point and where the class is not t he same as the idx-th data pt
                _b = pdist[
                    idx, lookup
                ].mean()  # dont need to exclude the diagonal entry because we know that its already not in lookup
                cluster_b_dict[clust_id] = _b

            b = cluster_b_dict[
                min(cluster_b_dict, key=cluster_b_dict.get)
            ]  # get the cluster ID for which the average distances are smallest

            scores[idx] = (b - a) / max(a, b)
        return scores  # will be one score per data point
