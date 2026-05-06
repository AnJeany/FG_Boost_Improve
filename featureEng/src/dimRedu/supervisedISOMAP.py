import warnings
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, shortest_path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.graph import _fix_connected_components
from sklearn.utils.validation import check_is_fitted


class SupervisedIsomap(TransformerMixin, BaseEstimator):
    """Supervised Isomap Embedding.

    Non-linear dimensionality reduction through Isometric Mapping with supervision.

    Parameters
    ----------
    n_neighbors : int or None, default=5
        Number of neighbors to consider for each point. If `n_neighbors` is an int,
        then `radius` must be `None`.

    radius : float or None, default=None
        Limiting distance of neighbors to return. If `radius` is a float,
        then `n_neighbors` must be set to `None`.

    n_components : int, default=2
        Number of coordinates for the manifold.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        'auto' : Attempt to choose the most efficient solver
        for the given problem.

        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.

        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.

    tol : float, default=0
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : int, default=None
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : {'auto', 'FW', 'D'}, default='auto'
        Method to use in finding shortest path.

        'auto' : attempt to choose the best algorithm automatically.

        'FW' : Floyd-Warshall algorithm.

        'D' : Dijkstra's algorithm.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    metric : str, or callable, default="minkowski"
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    """

    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=None,
        n_components=2,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        path_method="auto",
        neighbors_algorithm="auto",
        n_jobs=None,
        metric="minkowski",
        p=2,
        metric_params=None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _fit_transform(self, X, y):
        if self.n_neighbors is not None and self.radius is not None:
            raise ValueError(
                "Both n_neighbors and radius are provided. Use"
                f" SupervisedIsomap(radius={self.radius}, n_neighbors=None) if intended to use"
                " radius-based neighbors"
            )

        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm=self.neighbors_algorithm,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        # import time
        # t0 = time.time()
        self.nbrs_.fit(X)
        # t1 = time.time()
        # print('NearestNeighbors', t1- t0)
        self.n_features_in_ = self.nbrs_.n_features_in_
        if hasattr(self.nbrs_, "feature_names_in_"):
            self.feature_names_in_ = self.nbrs_.feature_names_in_

        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        ).set_output(transform="default")



        if self.n_neighbors is not None:
            nbg = kneighbors_graph(
                self.nbrs_,
                self.n_neighbors,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                mode="distance",
                n_jobs=self.n_jobs,
            )
        else:
            nbg = radius_neighbors_graph(
                self.nbrs_,
                radius=self.radius,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                mode="distance",
                n_jobs=self.n_jobs,
            )
        # t2 = time.time()
        # print('ngb', t2- t1)
        # Incorporate label information into the distance matrix
        adj_matrix = nbg.toarray()
        # for i in range(len(X)):
        #     for j in range(len(X)):
        #         if y[i] == y[j]:
        #             adj_matrix[i, j] *= 0.5  # Increase similarity within the same class
        #         else:
        #             adj_matrix[i, j] *= 2.0  # Decrease similarity between different classes
        # Create a mask for elements where y[i] == y[j]
        # Reshape y to be a 1D array if it's not already
        y = y.reshape(-1)
        same_label_mask = y[:, np.newaxis] == y
        # Apply the operations
        adj_matrix[same_label_mask] *= 0.5
        adj_matrix[~same_label_mask] *= 2.0

        nbg = adj_matrix
        # t3 = time.time()
        # print('assign array', t3- t2)
        # Compute the number of connected components, and connect the different
        # components to be able to compute a shortest path between all pairs
        # of samples in the graph.
        # Similar fix to cluster._agglomerative._fix_connectivity.
        n_connected_components, labels = connected_components(nbg)
        # t4 = time.time()
        # print('connected_components ',  t4- t3)
        if n_connected_components > 1:
            if self.metric == "precomputed" and issparse(X):
                raise RuntimeError(
                    "The number of connected components of the neighbors graph"
                    f" is {n_connected_components} > 1. The graph cannot be "
                    "completed with metric='precomputed', and SupervisedIsomap cannot be"
                    "fitted. Increase the number of neighbors to avoid this "
                    "issue, or precompute the full distance matrix instead "
                    "of passing a sparse neighbors graph."
                )
            warnings.warn(
                (
                    "The number of connected components of the neighbors graph "
                    f"is {n_connected_components} > 1. Completing the graph to fit"
                    " SupervisedIsomap might be slow. Increase the number of neighbors to "
                    "avoid this issue."
                ),
                stacklevel=2,
            )

            # use array validated by NearestNeighbors
            nbg = _fix_connected_components(
                X=self.nbrs_._fit_X,
                graph=nbg,
                n_connected_components=n_connected_components,
                component_labels=labels,
                mode="distance",
                metric=self.nbrs_.effective_metric_,
                **self.nbrs_.effective_metric_params_,
            )

        self.dist_matrix_ = shortest_path(nbg, method=self.path_method, directed=False)
        # t5 = time.time()
        # print('shortest_path', t5- t4)
        if self.nbrs_._fit_X.dtype == np.float32:
            self.dist_matrix_ = self.dist_matrix_.astype(
                self.nbrs_._fit_X.dtype, copy=False
            )

        G = self.dist_matrix_**2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)
        self._n_features_out = self.embedding_.shape[1]

    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float
            Reconstruction error.

        Notes
        -----
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_**2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.eigenvalues_
        return np.sqrt(np.sum(G_center**2) - np.sum(evals**2)) / G.shape[0]

    def fit(self, X, y):
        """Compute the embedding vectors for data X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, sparse matrix, precomputed tree, or NearestNeighbors
            object.

        y : array-like, shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        self._fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,)
            Target labels.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        self._fit_transform(X, y)
        return self.embedding_

    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features)
            If neighbors_algorithm='precomputed', X is assumed to be a
            distance matrix or a sparse graph of shape
            (n_queries, n_samples_fit).

        Returns
        -------
        X_new : array-like, shape (n_queries, n_components)
            X transformed in the new space.
        """
        check_is_fitted(self)
        if self.n_neighbors is not None:
            distances, indices = self.nbrs_.kneighbors(X, return_distance=True)
        else:
            distances, indices = self.nbrs_.radius_neighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to
        # training data via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:

        n_samples_fit = self.nbrs_.n_samples_fit_
        n_queries = distances.shape[0]

        if hasattr(X, "dtype") and X.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64

        G_X = np.zeros((n_queries, n_samples_fit), dtype)
        for i in range(n_queries):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] + distances[i][:, None], 0)

        G_X **= 2
        G_X *= -0.5

        return self.kernel_pca_.transform(G_X)

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}
    def get_name(self):
        return 'SupervisedIsomap'
