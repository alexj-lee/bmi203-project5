""" 
        Functions and classes to implement a basic K-means clustering. We use K-means ++ to initialize clusters 
        and a within-cluster mean square error computation to end optimization.

        We also implement a simple silhouette metric to evaluate clusters. 
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import make_clusters, plot_clusters, plot_multipanel

__version__ = "0.1"
