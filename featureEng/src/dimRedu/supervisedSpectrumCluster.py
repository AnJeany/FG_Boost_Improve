import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg

from sklearn.base import BaseEstimator, _fit_context
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array, check_random_state, check_symmetric
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import laplacian as csgraph_laplacian
from sklearn.utils.fixes import parse_version, sp_version


def _graph_connected_component(graph, node_id):
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[[i], :].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    if sparse.issparse(graph):
        accept_large_sparse = sp_version >= parse_version("1.11.3")
        graph = check_array(graph, accept_sparse=True, accept_large_sparse=accept_large_sparse)
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    n_nodes = laplacian.shape[0]
    if not sparse.issparse(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            laplacian = laplacian.tocsr()
    return laplacian


@validate_params(
    {
        "adjacency": ["array-like"],#, "sparse matrix"
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
        "random_state": ["random_state"],
        "eigen_tol": [Interval(Real, 0, None, closed="left"), StrOptions({"auto"})],
        "norm_laplacian": ["boolean"],
        "drop_first": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def spectral_embedding(
    adjacency,
    *,
    n_components=8,
    eigen_solver=None,
    random_state=None,
    eigen_tol="auto",
    norm_laplacian=True,
    drop_first=True,
):
    random_state = check_random_state(random_state)

    return _spectral_embedding(
        adjacency,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=norm_laplacian,
        drop_first=drop_first,
    )


def _spectral_embedding(
    adjacency,
    *,
    n_components=8,
    eigen_solver=None,
    random_state=None,
    eigen_tol="auto",
    norm_laplacian=True,
    drop_first=True,
):
    adjacency = check_symmetric(adjacency)

    if eigen_solver == "amg":
        try:
            from pyamg import smoothed_aggregation_solver
        except ImportError as e:
            raise ValueError(
                "The eigen_solver was set to 'amg', but pyamg is not available."
            ) from e

    if eigen_solver is None:
        eigen_solver = "arpack"

    n_nodes = adjacency.shape[0]
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )
    if (
        eigen_solver == "arpack"
        or eigen_solver != "lobpcg"
        and (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)
    ):
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        try:
            tol = 0 if eigen_tol == "auto" else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            laplacian = check_array(
                laplacian, accept_sparse="csr", accept_large_sparse=False
            )
            _, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = "lobpcg"
            laplacian *= -1

    elif eigen_solver == "amg":
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        if hasattr(sparse, "csr_array") and isinstance(laplacian, sparse.csr_array):
            laplacian = sparse.csr_matrix(laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse="csr"))
        laplacian -= diag_shift

        M = ml.aspreconditioner()
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)

        tol = None if eigen_tol == "auto" else eigen_tol
        _, diffusion_map = lobpcg(laplacian, X, M=M, tol=tol, largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError

    if eigen_solver == "lobpcg":
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        if n_nodes < 5 * n_components + 1:
            if sparse.issparse(laplacian):
                laplacian = laplacian.toarray()
            _, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            X = random_state.standard_normal(
                size=(laplacian.shape[0], n_components + 1)
            )
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            tol = None if eigen_tol == "auto" else eigen_tol
            _, diffusion_map = lobpcg(
                laplacian, X, tol=tol, largest=False, maxiter=2000
            )
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


class SupervisedSpectralEmbedding(BaseEstimator):
    def __init__(
        self,
        *,
        n_components=2,
        affinity="rbf",
        gamma=None,
        random_state=None,
        eigen_solver=None,
        n_neighbors=None,
        eigen_tol="auto",
        norm_laplacian=True,
        drop_first=True,
    ):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.norm_laplacian = norm_laplacian
        self.drop_first = drop_first

    def fit_transform(self, X, y=None):
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])
        random_state = check_random_state(self.random_state)

        self.X_train_ = X
        self.y_train_ = y

        self.affinity_matrix_ = self._compute_affinity_matrix(X, y)

        self.embedding_ = spectral_embedding(
            self.affinity_matrix_,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            eigen_tol=self.eigen_tol,
            norm_laplacian=self.norm_laplacian,
            drop_first=self.drop_first,
        )

        return self.embedding_

    def _compute_affinity_matrix(self, X, y=None):
        if self.affinity == "precomputed":
            return X
        elif self.affinity == "rbf":
            gamma = self.gamma
            if gamma is None:
                gamma = 1.0 / X.shape[1]
            affinity_matrix = rbf_kernel(X, gamma=gamma)
        elif self.affinity == "nearest_neighbors":
            n_neighbors = self.n_neighbors
            if n_neighbors is None:
                n_neighbors = max(int(np.log(X.shape[0])) + 1, 2)
            knn_graph = kneighbors_graph(X, n_neighbors, include_self=True)
            affinity_matrix = 0.5 * (knn_graph + knn_graph.T)
        else:
            raise ValueError(
                "Unknown affinity type. Expected 'precomputed', 'rbf', or 'nearest_neighbors', got {0}.".format(
                    self.affinity
                )
            )

        if y is not None:
            unique_labels = np.unique(y)
            for label in unique_labels:
                indices = np.where(y == label)[0]
                affinity_matrix[indices[:, None], indices] += 1

        return affinity_matrix

    def transform(self, X):
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])

        if self.affinity == "rbf":
            gamma = self.gamma
            if gamma is None:
                gamma = 1.0 / self.X_train_.shape[1]
            K_test = rbf_kernel(X, self.X_train_, gamma=gamma)
        elif self.affinity == "nearest_neighbors":
            n_neighbors = self.n_neighbors
            if n_neighbors is None:
                n_neighbors = max(int(np.log(self.X_train_.shape[0])) + 1, 2)
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(self.X_train_)
            K_test = nn.kneighbors(X, return_distance=False)
            affinity_matrix = kneighbors_graph(self.X_train_, n_neighbors, include_self=True)
            K_test = 0.5 * (affinity_matrix[K_test] + affinity_matrix[K_test].T)
        else:
            raise ValueError("Unknown affinity type. Expected 'rbf' or 'nearest_neighbors'.")

        return np.dot(K_test, self.embedding_)
    def get_name(self):
        return 'SupervisedSpectralEmbedding'

# from sklearn.base import BaseEstimator, _fit_context
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.neighbors import NearestNeighbors, kneighbors_graph
# from sklearn.utils import check_array, check_random_state, check_symmetric
# from sklearn.utils._arpack import _init_arpack_v0
# from sklearn.utils.extmath import _deterministic_vector_sign_flip
# from sklearn.utils.fixes import laplacian as csgraph_laplacian
# from sklearn.utils.fixes import parse_version, sp_version
# import warnings
# from scipy import sparse
# from scipy.linalg import eigh
# from scipy.sparse.csgraph import connected_components
# from scipy.sparse.linalg import eigsh, lobpcg
# import numpy as np
#
# class SupervisedSpectralEmbedding(BaseEstimator):
#     def __init__(self, n_components=2, affinity='nearest_neighbors', gamma=None,
#                  random_state=None, eigen_solver=None, eigen_tol='auto',
#                  n_neighbors=None, n_jobs=None):
#         self.n_components = n_components
#         self.affinity = affinity
#         self.gamma = gamma
#         self.random_state = random_state
#         self.eigen_solver = eigen_solver
#         self.eigen_tol = eigen_tol
#         self.n_neighbors = n_neighbors
#         self.n_jobs = n_jobs
#
#     def fit(self, X, y=None):
#         self.fit_transform(X, y)
#         return self
#
#     def fit_transform(self, X, y=None):
#         random_state = check_random_state(self.random_state)
#         self.embedding_ = self._spectral_embedding(X, y, random_state)
#         return self.embedding_
#
#     def _spectral_embedding(self, X, y, random_state):
#         affinity_matrix = self._compute_affinity_matrix(X, y)
#         adjacency = check_symmetric(affinity_matrix)
#
#         if self.eigen_solver is None:
#             eigen_solver = "arpack"
#         else:
#             eigen_solver = self.eigen_solver
#
#         n_nodes = adjacency.shape[0]
#         n_components = self.n_components + 1
#
#         laplacian, dd = csgraph_laplacian(adjacency, normed=True, return_diag=True)
#
#         if eigen_solver == "arpack" or (eigen_solver != "lobpcg" and
#                                         (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)):
#             laplacian = self._set_diag(laplacian, 1, True)
#             v0 = _init_arpack_v0(laplacian.shape[0], random_state)
#             laplacian = check_array(laplacian, accept_sparse="csr")
#             _, diffusion_map = eigsh(laplacian, k=n_components, sigma=1.0, which="LM", v0=v0)
#             embedding = diffusion_map.T[n_components::-1]
#             embedding = embedding / dd
#         else:
#             raise ValueError("Only 'arpack' eigen_solver is implemented in this example.")
#
#         embedding = _deterministic_vector_sign_flip(embedding)
#         return embedding[1:self.n_components+1].T
#
#     def _set_diag(self, laplacian, value, norm_laplacian):
#         n_nodes = laplacian.shape[0]
#         if not sparse.issparse(laplacian):
#             if norm_laplacian:
#                 laplacian.flat[:: n_nodes + 1] = value
#         else:
#             laplacian = laplacian.tocoo()
#             if norm_laplacian:
#                 diag_idx = laplacian.row == laplacian.col
#                 laplacian.data[diag_idx] = value
#             n_diags = np.unique(laplacian.row - laplacian.col).size
#             if n_diags <= 7:
#                 laplacian = laplacian.todia()
#             else:
#                 laplacian = laplacian.tocsr()
#         return laplacian
#
#     def _compute_affinity_matrix(self, X, y):
#         if callable(self.affinity):
#             return self.affinity(X, y)
#
#         if self.affinity == 'nearest_neighbors':
#             connectivity = kneighbors_graph(X, self.n_neighbors, include_self=True, n_jobs=self.n_jobs)
#             return 0.5 * (connectivity + connectivity.T)
#         elif self.affinity == 'rbf':
#             gamma = self.gamma
#             if gamma is None:
#                 gamma = 1.0 / X.shape[1]
#             return rbf_kernel(X, gamma=gamma)
#         elif self.affinity == 'precomputed':
#             return X
#         elif self.affinity == 'precomputed_nearest_neighbors':
#             connectivity = X
#             return 0.5 * (connectivity + connectivity.T)
#         else:
#             raise ValueError(f"Unknown affinity type '{self.affinity}'")
