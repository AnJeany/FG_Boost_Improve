import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh

# landmarks index is randomly selected.
class SupervisedFastMVU:
    def __init__(self, n_components=2, n_landmarks=20):
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.landmarks = None
        self.B = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.landmarks = X[np.random.choice(n_samples, self.n_landmarks, replace=False)]
        dist_matrix = pairwise_distances(X, self.landmarks)
        L_dist_matrix = pairwise_distances(self.landmarks)

        for i in range(n_samples):
            for j in range(self.n_landmarks):
                if y[i] == y[j % n_samples]:
                    dist_matrix[i, j] *= 0.5  # Increase similarity within the same class
                else:
                    dist_matrix[i, j] *= 2.0  # Decrease similarity between different classes

        for i in range(self.n_landmarks):
            for j in range(self.n_landmarks):
                if y[i % n_samples] == y[j % n_samples]:
                    L_dist_matrix[i, j] *= 0.5
                else:
                    L_dist_matrix[i, j] *= 2.0

        J = np.eye(self.n_landmarks) - (1/self.n_landmarks) * np.ones((self.n_landmarks, self.n_landmarks))
        K = -0.5 * J @ L_dist_matrix @ J

        eigenvalues, eigenvectors = eigh(K)
        idx = np.argsort(-eigenvalues)[:self.n_components]
        eigenvalues[eigenvalues < 0.0] = 0.0
        L_embedding = eigenvectors[:, idx] @ np.diag(np.sqrt(eigenvalues[idx]))

        self.B = np.linalg.pinv(dist_matrix.T @ dist_matrix)  @ L_embedding

        return self

    def transform(self, X):
        dist_matrix = pairwise_distances(X, self.landmarks)
        return dist_matrix @ self.B

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    def get_name(self):
        return 'SupervisedFastMVU'

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
    s_lle = SupervisedFastMVU(n_components=2, n_landmarks=20)

    # Fit and transform the data
    X_transformed = s_lle.fit_transform(X, y)
    print(X_transformed.shape)