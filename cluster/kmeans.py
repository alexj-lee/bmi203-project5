from cmath import inf
from re import S
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance


class KMeans:
    def __init__(
        self, k: int, metric: str = "euclidean", tol: float = 1e-6, max_iter: int = 100
    ):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None

        self._validate_init()

    def _validate_init(self):
        metric = distance.__dict__.get(self.metric)

        if metric is None:
            raise ValueError(
                'Argument "metric" must be a valid pairwise distance metric found in scipy.spatial.distance.'
            )
        elif callable(metric) is False:
            raise ValueError(
                f"Metric must be a function found in scipy.spatial.distance. Found object is of type {type(metric)}"
            )

        self.distance = metric

        if isinstance(self.k, (float, int)) is False:
            raise TypeError("Argument k must be numeric.")

        elif self.k <= 1:
            raise ValueError(f"Argument k must be larger than or equal to 2.")

        try:
            self.tol = float(self.tol)
        except:
            raise TypeError(f"Argument {self.tol} must be a numeric.")

        if np.sign(self.tol) == -1 or np.greater_equal(self.tol, 0) is False:
            raise ValueError("Argument tol must be positive.")

        try:
            self.max_iter = int(self.max_iter)
        except:
            raise TypeError(f"Argument {self.max_iter} must be a numeric.")

        if self.max_iter <= 0:
            raise ValueError(f"Argument max_iter must be at least 1.")

    def _init_pdist_matrix(self, x: np.ndarray):
        if self.k <= len(x):
            raise ValueError(
                "Clustering input matrix has fewer or equal data points to k; clustering will not be meaningful. Please increase number of data points or reduce k."
            )

        if x.ndim != 2:
            raise ValueError("Input matrix must be a 2D matrix.")

        pass

    def _init_centroids(self):
        """
        Initialize centroids of the k-means clustering with the kmeans ++ initialization algorithm.
        [https://en.wikipedia.org/wiki/K-means%2B%2B#Improved_initialization_algorithm]

        1. Chose a random data point in x, x_i in X (whole dataset)
        2. For all {j | x_j != x_i, x_j in X}, calculate distance D(x_j, x_i) between x_j and x_i
        3. Chose a new data point x_k as a new centroid, chosing x_k randomly with probability proportional to D(x_k, x_i)
        4. Repeat until we hit the desired number of clusters.

        Returns:
            [np.ndarray]: cluster centers, with (n_centroids, n_features) dimensions
        """

        if getattr(self, "x") is None:
            raise Exception("You need to run fit before you run _init_centroids.")

        centroids = np.zeros((self.k, self.num_features))

        distances = np.full(self.num_data_points, np.inf).tolist()
        probabilities = (np.ones(self.num_data_points) / self.num_data_points).tolist()

        indices = np.arange(len(self.x)).tolist()

        for iter_idx in range(self.k):
            centroid_idx = np.random.choice(indices, 1, p=probabilities).item()
            centroid_coords = self.x[centroid_idx]
            centroids[iter_idx] = centroid_coords

            for lst in (indices, probabilities, distances):
                del lst[centroid_idx]

            for array_idx, pt_idx in enumerate(indices):
                distance = self.distance(centroid_coords, self.x[pt_idx])

                if (
                    distance < distances[array_idx]
                ):  # only replace distance if its to the closest centroid
                    distances[array_idx] = distance

            distance_sum = sum(distances)
            probabilities = [distance / distance_sum for distance in distances]

        return centroids

    def _cluster_cdist(self, x: np.ndarray, centroids: np.ndarray):
        return distance.cdist(x, centroids)

    def _get_error(self, centroids: np.ndarray):
        cdist = self._cluster_cdist(self.x, centroids)
        best_centroids = cdist.argmin(1)

        score = 0
        for idx in range(self.k):
            score += cdist[best_centroids == idx].sum()

        return score

    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self.x = mat
        self.num_data_points, self.num_features = mat.shape
        centroids = self._init_centroids()

        tol = np.inf
        iteration = 0

        score_prev = self._get_error(centroids)
        score_current = score_prev
        delta = abs(score_current - score_prev)

        print("starting", delta < self.tol, iteration < self.max_iter)

        while iteration < self.max_iter:
            cdist = self._cluster_cdist(self.x, centroids)
            # return cdist, centroids
            best_centroids = cdist.argmin(1)

            score_prev = score_current
            score_current = 0

            for idx in range(self.k):
                new_centroid_coord = self.x[best_centroids == idx].mean(
                    0
                )  # is this right?
                centroids[idx] = new_centroid_coord

            for idx in range(self.k):
                new_cdist = self._cluster_cdist(self.x, centroids)

                _score = new_cdist[best_centroids == idx].sum()
                score_current += _score

            iteration += 1
            delta = abs(score_current - score_prev)

            if delta < self.tol:
                break

        self.centroids = centroids
        self._labels = best_centroids
        self._error = score_current
        return centroids, best_centroids

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if self.centroids is None:
            raise Exception("K-means hasn't been fit yet.  Please run object.fit()")

        if mat.shape[1] != self.num_features:
            pass

        cdist = self._cluster_cdist(mat, self.centroids)
        cluster_labels = cdist.argmin(1)
        return cluster_labels

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        if getattr(self, "final_error", None) is None:
            cdist = distance.cdist(self.x, self.centroids, "euclidean")
            best_centroids = cdist.argmin(1)
            scores = [
                cdist[best_centroids == centroid_idx].sum()
                for centroid_idx in range(self.k)
            ]
            scores = sum(scores) / self.num_data_points
            self.final_error = scores

        return self.final_error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
