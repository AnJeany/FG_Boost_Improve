
"""
Multi-dimensional Scaling (MDS).
"""

# author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD

import warnings
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs

from sklearn.base import BaseEstimator, _fit_context
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array, check_random_state, check_symmetric
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

'''https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/manifold/_mds.py
'''

def _smacof_single(
    dissimilarities,
    metric=True,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    normalized_stress=False,
):

    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.uniform(size=n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    old_stress = None
    ir = IsotonicRegression()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt(
                (n_samples * (n_samples - 1) / 2) / (disparities**2).sum()
            )

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        if normalized_stress:
            stress = np.sqrt(stress / ((disparities.ravel() ** 2).sum() / 2))
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1.0 / n_samples * np.dot(B, X)

        dis = np.sqrt((X**2).sum(axis=1)).sum()
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print("breaking at iteration %d with stress %s" % (it, stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1




@validate_params(
    {
        "dissimilarities": ["array-like"],
        "metric": ["boolean"],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "init": ["array-like", None],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "eps": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "return_n_iter": ["boolean"],
        "normalized_stress": ["boolean", StrOptions({"auto"})],
    },
    prefer_skip_nested_validation=True,
)
def smacof(
    dissimilarities,
    *,
    metric=True,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    verbose=0,
    eps=1e-3,
    random_state=None,
    return_n_iter=False,
    normalized_stress="auto",
):


    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if normalized_stress == "auto":
        normalized_stress = not metric

    if normalized_stress and metric:
        raise ValueError(
            "Normalized stress is not supported for metric MDS. Either set"
            " `normalized_stress=False` or use `metric=False`."
        )
    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
                normalized_stress=normalized_stress,
            )
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
                normalized_stress=normalized_stress,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress

class FastSupervisedMDS(BaseEstimator):
    def __init__(self, n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=1e-3,
                 n_jobs=None, random_state=None, dissimilarity="euclidean", normalized_stress=False,
                 n_landmarks=None):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.eps = eps
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        self.normalized_stress = normalized_stress
        self.n_landmarks = n_landmarks

    def fit(self, X, y=None, init=None):
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)

        if self.n_landmarks is not None and self.n_landmarks < X.shape[0]:
            indices = random_state.choice(X.shape[0], self.n_landmarks, replace=False)
            self.landmark_indices_ = indices
            landmark_dissimilarities = self.dissimilarity_matrix_[np.ix_(indices, indices)]
            X_landmarks, stress, _ = _smacof_single(
                landmark_dissimilarities, metric=self.metric, n_components=self.n_components, init=init,
                max_iter=self.max_iter, verbose=self.verbose, eps=self.eps, random_state=random_state,
                normalized_stress=self.normalized_stress
            )
            self.landmarks_ = X_landmarks
            self.landmark_original_ = X[indices]
            full_distances = self.dissimilarity_matrix_[:, indices]
            self.embedding_ = self._transform_landmarks(X_landmarks, full_distances)
        else:
            self.embedding_, self.stress_, self.n_iter_ = _smacof_single(
                self.dissimilarity_matrix_, metric=self.metric, n_components=self.n_components, init=init,
                max_iter=self.max_iter, verbose=self.verbose, eps=self.eps, random_state=random_state,
                normalized_stress=self.normalized_stress
            )
        return self.embedding_

    def transform(self, X):
        check_is_fitted(self, 'landmarks_')
        X = check_array(X, accept_sparse='csr')
        if self.dissimilarity == "precomputed":
            distances = X[:, self.landmark_indices_]
        elif self.dissimilarity == "euclidean":
            distances = euclidean_distances(X, self.landmark_original_)
        return self._transform_landmarks(self.landmarks_, distances)

    def _transform_landmarks(self, X_landmarks, full_distances):
        n_samples = full_distances.shape[0]
        n_landmarks = X_landmarks.shape[0]
        B = np.zeros((n_samples, n_landmarks))
        for i in range(n_samples):
            d = full_distances[i]
            B[i] = -d / np.linalg.norm(d)
        return np.dot(B, X_landmarks)

    def get_name(self):
        return 'SupervisedMDS'