from cluster import make_clusters
import numpy as np
import pathlib

test_clustering, test_labels = make_clusters(scale=1, k=3, n=1000, m=100, seed=0)

full_data = np.c_[test_clustering, test_labels]

np.save(pathlib.Path(__file__).resolve().parent / "test_data.npy", full_data)

test_clustering, test_labels = cluster.make_clusters(scale=7, k=3, n=200, m=3, seed=0)
np.save(pathlib.Path(__file__).resolve().parent / "test_data_mixed.npy", full_data)
