from cmath import inf
from re import S
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None

        self._validate_init()

    def _validate_init(self):
        # check if k is a number that is greater than 1
        if isinstance(self.k, (float, int)) is False:
            raise TypeError("Argument k must be numeric.")

        elif self.k <= 1:
            raise ValueError(f"Argument k must be larger than or equal to 2.")

        # same for toleranec, but tol just needs to be greater than 0
        try:
            self.tol = float(self.tol)
        except:
            raise TypeError(f"Argument {self.tol} must be a numeric.")

        if np.sign(self.tol) == -1 or np.greater(self.tol, 0) is False:
            raise ValueError("Argument tol must be positive.")

        # max iter must be convertable to an integer
        try:
            self.max_iter = int(self.max_iter)
        except:
            raise TypeError(f"Argument {self.max_iter} must be a numeric.")

        if self.max_iter <= 0:
            raise ValueError(f"Argument max_iter must be at least 1.")

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

        centroids = np.zeros((self.k, self.num_features))  # allocate the clusters

        distances = np.full(
            self.num_data_points, np.inf
        ).tolist()  # get self.num_data_points long vector of infinities
        probabilities = (
            np.ones(self.num_data_points) / self.num_data_points
        ).tolist()  # initialize probabilities to 1/n

        # candidate indices for centroids selection
        indices = np.arange(
            len(self.x)
        ).tolist()  # convenient to use indices here to index the data so we can remove the data points that werer selected as centroids for knn+

        for iter_idx in range(self.k):
            centroid_idx = np.random.choice(
                indices, 1, p=probabilities
            ).item()  # pick a data point idx
            centroid_coords = self.x[centroid_idx]
            centroids[iter_idx] = centroid_coords  # make that the iter_idx-th centroid

            for lst in (indices, probabilities, distances):
                del lst[
                    centroid_idx
                ]  # remove the centroid_idx-th data point from the candidate list

            for array_idx, pt_idx in enumerate(
                indices
            ):  # get the distance between the centroid we just chose and the rest of the data points
                distance = np.power(dist.euclidean(centroid_coords, self.x[pt_idx]), 2)

                if (
                    distance < distances[array_idx]
                ):  # only replace distance if its to the closest centroid
                    distances[array_idx] = distance

            distance_sum = sum(distances)
            probabilities = [
                distance / distance_sum for distance in distances
            ]  # probabilities are normalized over distance

        return centroids

    def _cluster_cdist(self, x: np.ndarray, centroids: np.ndarray):
        # compute the pairwise distance between the data points (x) and centroids
        return dist.cdist(x, centroids)

    def _get_error(self, centroids: np.ndarray):
        """
        Returns the distance between the data point and the closest centroid as an average over all data points.

        Args:
            centroids (np.ndarray): Centroids as a (k, nfeatures) matrix

        Returns:
            float: the distances between the best clusters and the centroids

        """

        cdist = self._cluster_cdist(self.x, centroids)
        cdist_sq = np.power(cdist, 2)

        score = np.amin(cdist_sq, axis=1).mean()

        return cdist, score

    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        self.x = mat
        if mat.ndim != 2:
            raise ValueError("Input matrix x must be a 2D matrix.")

        if len(mat) < self.k:
            raise ValueError(
                f"We can't fit k means with k={self.k} with {len(mat)} data points."
            )

        self.num_data_points, self.num_features = mat.shape
        centroids = (
            self._init_centroids()
        )  # use knn++ to initialize centroids (ncentroids, nfeatures)

        # initialize variables to help us figure out how long to loop
        tol = np.inf
        iteration = 0

        cdist, score_prev = self._get_error(
            centroids
        )  # how far are the data points overall from their closest centroid as a single number
        score_current = score_prev
        delta = abs(score_current - score_prev)

        # we're just going to go around on loops and each time we'll find the cloosest centroid to each point
        # and then average the

        best_centroids = None
        while iteration < self.max_iter:

            # cdist = self._cluster_cdist(self.x, centroids)
            best_centroids = cdist.argmin(
                1
            )  # will be the best centroid's index for e/ data point as a n_data_pts long vect

            score_prev = score_current
            #            score_current = 0

            for idx in range(self.k):
                new_centroid_coord = self.x[best_centroids == idx].mean(
                    0
                )  # average together the data points for the data points assigned to idx-th cluster
                centroids[idx] = new_centroid_coord

            cdist, score_current = self._get_error(centroids)

            iteration += 1
            delta = abs(score_current - score_prev)

            if delta < self.tol:
                break

        self.centroids = centroids
        self._labels = best_centroids
        self.final_error = score_current
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
            raise ValueError(
                f"K-means must have {self.num_features} features for input data with current fit."
            )

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
            if getattr(self, "centroids", None) is None:
                raise RuntimeError(
                    "You need to fit the centroids before you get the error value."
                )
        else:
            return self.final_error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        if getattr(self, "centroids", None) is None:
            raise RuntimeError(
                "You need to fit the centroids before you can return them."
            )

        return self.centroids
