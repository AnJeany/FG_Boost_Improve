from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh, qr, solve, svd
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh

from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

def barycenter_weights(X, Y, indices, reg=1e-3):
    X = check_array(X, dtype=FLOAT_DTYPES)
    Y = check_array(Y, dtype=FLOAT_DTYPES)
    indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, assume_a="pos")
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))

def null_space(M, k, k_skip=1, eigen_solver="arpack", tol=1e-6, max_iter=100, random_state=None):
    if eigen_solver == "auto":
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = "arpack"
        else:
            eigen_solver = "dense"

    if eigen_solver == "arpack":
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            eigen_values, eigen_vectors = eigsh(
                M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0
            )
        except RuntimeError as e:
            raise ValueError(
                "Error in determining null-space with ARPACK. Error message: "
                "'%s'. Note that eigen_solver='arpack' can fail when the "
                "weight matrix is singular or otherwise ill-behaved. In that "
                "case, eigen_solver='dense' is recommended. See online "
                "documentation for more information." % e
            ) from e

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == "dense":
        if hasattr(M, "toarray"):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, subset_by_index=(k_skip, k + k_skip - 1), overwrite_a=True
        )
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)

def _supervised_locally_linear_embedding(
    X,
    y,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples,  but n_samples = %d, n_neighbors = %d"
            % (N, n_neighbors)
        )

    M_sparse = eigen_solver != "dense"

    if method == "standard":
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
        )

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I

    elif method == "hessian":
        dp = n_components * (n_components + 1) // 2

        if n_neighbors <= n_components + dp:
            raise ValueError(
                "for method='hessian', n_neighbors must be "
                "greater than "
                "[n_components * (n_components + 3) / 2]"
            )

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1

        M = np.zeros((N, N), dtype=np.float64)

        use_svd = n_neighbors > d_in

        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)

            # build Hessian estimator
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]

            Yi[:, 1 : 1 + n_components] = U[:, :n_components]

            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j : j + n_components - k] = U[:, k : k + 1] * U[:, k:n_components]
                j += n_components - k

            Q, R = qr(Yi)

            w = Q[:, n_components + 1 :]
            S = w.sum(0)

            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)

        if M_sparse:
            M = csr_matrix(M)

    elif method == "modified":
        if n_neighbors < n_components:
            raise ValueError("modified LLE requires n_neighbors >= n_components")

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        # find the eigenvectors and eigenvalues of each local covariance
        # matrix. We want V[i] to be a [n_neighbors x n_neighbors] matrix,
        # where the columns are eigenvectors
        V = np.zeros((N, n_neighbors, n_neighbors))
        nev = min(d_in, n_neighbors)
        evals = np.zeros([N, nev])

        # choose the most efficient way to find the eigenvectors
        use_svd = n_neighbors > d_in

        if use_svd:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                V[i], evals[i], _ = svd(X_nbrs, full_matrices=True)
            evals **= 2
        else:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                C_nbrs = np.dot(X_nbrs, X_nbrs.T)
                evi, vi = eigh(C_nbrs)
                evals[i] = evi[::-1]
                V[i] = vi[:, ::-1]

        # find regularized weights: this is like normal LLE.
        # because we've already computed the SVD of each covariance matrix,
        # it's faster to use this rather than np.linalg.solve
        reg = 1e-3 * evals.sum(1)

        tmp = np.dot(V.transpose(0, 2, 1), V.transpose(0, 2, 1))
        tmp = np.where(np.abs(tmp) < modified_tol, 0, tmp)
        for i in range(N):
            tmp[i].flat[:: n_neighbors + 1] += reg[i]

        w = np.empty((N, n_neighbors))
        for i in range(N):
            w[i] = solve(tmp[i], np.ones(n_neighbors), assume_a="pos")
            w[i] /= w[i].sum()

        W = csr_matrix(
            (w.ravel(), neighbors.ravel(), np.arange(0, N * n_neighbors + 1, n_neighbors))
        )

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I
    else:
        raise ValueError("unrecognized method '%s'" % method)

    if M_sparse:
        eigen_vectors, _ = null_space(
            M,
            n_components,
            k_skip=1,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        eigen_values, eigen_vectors = eigh(M, overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        eigen_vectors = eigen_vectors[:, index]
        index = np.argsort(np.abs(eigen_values))
        eigen_vectors = eigen_vectors[:, index]

    return eigen_vectors[:, :n_components], nbrs


class SupervisedLocallyLinearEmbedding(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "reg": [Interval(Real, 0.0, None, closed="left")],
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "method": [StrOptions({"standard", "hessian", "modified"})],
        "hessian_tol": [Interval(Real, 0.0, None, closed="left")],
        "modified_tol": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [Interval(Integral, None, None, closed="neither"), None],
    }

    def __init__(
        self,
        *,
        n_neighbors=5,
        n_components=2,
        reg=1e-3,
        eigen_solver="auto",
        tol=1e-6,
        max_iter=100,
        method="standard",
        hessian_tol=1e-4,
        modified_tol=1e-12,
        random_state=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _more_tags(self):
        return {"pairwise": True}

    def fit_transform(self, X, y=None):
        X = self._validate_data(X, dtype=FLOAT_DTYPES)
        random_state = check_random_state(self.random_state)

        self.embedding_, self.nbrs_ = _supervised_locally_linear_embedding(
            X,
            y,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            reg=self.reg,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            hessian_tol=self.hessian_tol,
            modified_tol=self.modified_tol,
            random_state=random_state,
            n_jobs=self.n_jobs,
        )

        return self.embedding_

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = check_array(X)
        ind = self.nbrs_.kneighbors(X, return_distance=False)

        weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new
        # return np.dot(weights, self.embedding_[ind])
    def get_name(self):
        return 'SupervisedLocallyLinearEmbedding'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris

    # Assuming SupervisedLocallyLinearEmbedding is defined in a module named supervised_lle
    # Load datasets
    data = load_iris()
    X = data.data
    y = data.target

    # Initialize the supervised LLE model
    s_lle = SupervisedLocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)

    # Fit and transform the data
    X_transformed = s_lle.fit_transform(X, y)
    print(X_transformed.shape)