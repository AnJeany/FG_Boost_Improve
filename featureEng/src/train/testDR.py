import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.kernel_approximation import RBFSampler, Nystroem, PolynomialCountSketch

from datasets import RFFGaussianProcess
from datasets import SupervisedIsomap
from datasets import SupervisedLDA
from datasets import SupervisedSpectralEmbedding
from datasets import SupervisedFastMVU
from datasets import SupervisedLocallyLinearEmbedding
import umap

from datasets import SupervisedTSNE


class my_testExgboost:
    # initial datasets
    def __init__(self, train_x=None, val_x=None, test_x=None, num_class=0, train_y=None, val_y=None, test_y=None):
        self.train_x = torch.tensor(train_x) if not isinstance(train_x, torch.Tensor) else train_x
        self.val_x = torch.tensor(val_x) if not isinstance(val_x, torch.Tensor) else val_x
        self.test_x = torch.tensor(test_x) if not isinstance(test_x, torch.Tensor) else test_x
        self.kClasses = num_class
        self.train_y = torch.tensor(train_y) if not isinstance(train_y, torch.Tensor) else train_y
        self.y_val = torch.tensor(val_y) if not isinstance(val_y, torch.Tensor) else val_y
        self.y_test = torch.tensor(test_y) if not isinstance(test_y, torch.Tensor) else test_y
        self._len = self.train_x.shape[0]
        self._dim = self.train_x.shape[1]

    def _sampling(self, train_probs):
        # Convert to PyTorch tensors if not already
        if not isinstance(train_probs, torch.Tensor):
            train_probs = torch.tensor(train_probs)
        if not isinstance(self.train_x, torch.Tensor):
            self.train_x = torch.tensor(self.train_x)
        if not isinstance(self.train_y, torch.Tensor):
            self.train_y = torch.tensor(self.train_y)

        # Sample the training data based on the sampling probabilities
        # Calculate uncertainty information (variance of probabilities)
        train_probs = train_probs / train_probs.sum(dim=1, keepdim=True)  # Normalize to sum to 1
        train_uncertainty = torch.var(train_probs, dim=1)
        # Normalize uncertainties to create sampling probabilities
        sampling_probabilities = train_uncertainty / train_uncertainty.sum()

        num_samples = int(self._len * 0.98)  # Adjust the number of samples if needed
        X_train_sampled = []
        y_train_sampled = []
        # Ensure stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=num_samples, random_state=42)
        for train_index, _ in sss.split(self.train_x.numpy(), self.train_y.numpy().ravel()):
            # Stratified sampling within each class
            unique_classes = self.train_y.unique()
            for cls in unique_classes:
                cls_indices = train_index[self.train_y[train_index].ravel()== cls]
                cls_probabilities = sampling_probabilities[cls_indices]
                cls_probabilities = cls_probabilities / cls_probabilities.sum()  # Normalize to sum to 1

                num_cls_samples = int(num_samples * (len(cls_indices) / len(train_index)))
                sampled_cls_indices = np.random.choice(cls_indices, size=num_cls_samples, replace=True,
                                                       p=cls_probabilities.numpy())

                X_train_sampled.append(self.train_x[sampled_cls_indices])
                y_train_sampled.append(self.train_y[sampled_cls_indices])

            X_train_sampled = torch.cat(X_train_sampled)
            y_train_sampled = torch.cat(y_train_sampled)

        return X_train_sampled, y_train_sampled

    def train(self, config, seed: int = 42) -> float:
        """Creates a XGB using cross-validation."""
 ##########        # data did not do centering processing
        # step 1: randomly initialize train_probs with the same shape as the model's output
        num_samples, num_classes = self.train_y.shape[0], len(np.unique(self.train_y))
        random_probs = np.random.rand(num_samples, num_classes)


        X_sampled, y_sampled = self._sampling(random_probs)
        # parameter affinity- decised what method you will use for affinity_matrix
        s_lda = SupervisedLDA()
        X_lle = s_lda.fit_transform(X_sampled, y_sampled.ravel())
        # Create and train the XGBoost model with early stopping
        model_test = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = s_lda.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_test.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = s_lda.transform(self.train_x)
        train_probs = model_test.predict_proba(train_data)
        cost_slle = 1 - metrics.accuracy_score(self.y_val, model_test.predict(val_data))
        print('cost_LDA', cost_slle)


        X_sampled, y_sampled = self._sampling(random_probs)
        # parameter affinity- decised what method you will use for affinity_matrix
        s_lda = SupervisedTSNE( n_components=3)
        X_lle = s_lda.fit_transform(X_sampled, y_sampled)
        # Create and train the XGBoost model with early stopping
        model_test = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = s_lda.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_test.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = s_lda.transform(self.train_x)
        train_probs = model_test.predict_proba(train_data)
        cost_slle = 1 - metrics.accuracy_score(self.y_val, model_test.predict(val_data))
        print('cost_ISOMAP', cost_slle)


        #
        X_sampled, y_sampled = self._sampling(random_probs)
        # parameter affinity- decised what method you will use for affinity_matrix
        s_test = SupervisedSpectralEmbedding( n_components=int(self._dim/2), n_neighbors=10, random_state=42)
        X_lle = s_test.fit_transform(X_sampled, y_sampled)
        # Create and train the XGBoost model with early stopping
        model_test = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = s_test.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_test.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = s_test.transform(self.train_x)
        train_probs = model_test.predict_proba(train_data)
        cost_slle = 1 - metrics.accuracy_score(self.y_val, model_test.predict(val_data))
        print('cost_ISOMAP', cost_slle)

        #
        X_sampled, y_sampled = self._sampling(random_probs)
        # step 2.1: dimensional reduction methods.
        s_test = SupervisedIsomap(n_neighbors=10, n_components=int(self._dim/2))
        X_lle = s_test.fit_transform(X_sampled, y_sampled)
        # Create and train the XGBoost model with early stopping
        model_test = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = s_test.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_test.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = s_test.transform(self.train_x)
        train_probs = model_test.predict_proba(train_data)
        cost_slle = 1 - metrics.accuracy_score(self.y_val, model_test.predict(val_data))
        print('cost_ISOMAP', cost_slle)


        # #-----------------------------------------------------------
        # step 2: sampling data and train a model and record the evaluation score
        X_sampled, y_sampled = self._sampling(random_probs)
        # step 2.1: dimensional reduction methods.
        s_lle = SupervisedLocallyLinearEmbedding(n_neighbors=10, n_components=int(self._dim/2), random_state=42)

        X_lle = s_lle.fit_transform(X_sampled, y_sampled)

        # Create and train the XGBoost model with early stopping
        model_slle = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)

        val_data = s_lle.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_slle.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = s_lle.transform(self.train_x)
        # Predict probabilities for the training and testing sets
        train_probs = model_slle.predict_proba(train_data)
        cost_slle = 1 - metrics.accuracy_score(self.y_val, model_slle.predict(val_data))
        print('cost_LLE', cost_slle)
        #-----------------------------------------------------------
        # step 3: goes to another round
        X_sampled, y_sampled = self._sampling(train_probs)
        mvu_mvu =  SupervisedFastMVU(n_components=int(self._dim/4), n_landmarks=20)
        X_lle = mvu_mvu.fit_transform(X_sampled, y_sampled)

        # Create and train the XGBoost model with early stopping
        model_mvu = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = mvu_mvu.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_mvu.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = mvu_mvu.transform(self.train_x)
        # Predict probabilities for the training and testing sets
        train_probs = model_mvu.predict_proba(train_data)
        cost_smvu = 1 - metrics.accuracy_score(self.y_val, model_mvu.predict(val_data))
        print('cost_SMVU', cost_smvu)
        #-----------------------------------------------------------
        # step 4: goes to another round
        X_sampled, y_sampled = self._sampling(train_probs)
        reducer = umap.UMAP(
            n_components=int(self._dim/8), n_neighbors=15, random_state=42, transform_seed=42, verbose=False
        )
        X_lle = reducer.fit_transform(X_sampled, y_sampled)

        # Create and train the XGBoost model with early stopping
        model_umap = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = reducer.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]

        model_umap.fit(X_lle, y= y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = reducer.transform(self.train_x)
        # Predict probabilities for the training and testing sets
        train_probs = model_umap.predict_proba(train_data)
        cost_umap = 1 - metrics.accuracy_score(self.y_val, model_umap.predict(val_data))
        print('cost_UMAP', cost_umap)
        #-----------------------------------------------------------
        # step 5: goes to another round
        X_sampled, y_sampled = self._sampling(train_probs)

        clf = RFFGaussianProcess(rff_dim=int(self._dim/4))
        X_lle = clf._get_rffs(X_sampled).T

        # Create and train the XGBoost model with early stopping
        model_randFourier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',  **config)
        val_data = clf._get_rffs(self.val_x).T
        eval_set = [ (val_data, self.y_val)]
        model_randFourier.fit(X_lle, y_sampled, early_stopping_rounds=10, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = clf._get_rffs(self.train_x).T
        # Predict probabilities for the training and testing sets
        train_probs = model_randFourier.predict_proba
        cost_rfGP = 1 - metrics.accuracy_score(self.y_val, model_randFourier.predict(val_data))
        print('cost_rfGP', cost_rfGP)
        #-----------------------------------------------------------
        # step 6: goes to another round
        train_probs = random_probs
        X_sampled, y_sampled = self._sampling(train_probs)

        rbf_feature = RBFSampler(gamma=1, random_state=1,n_components=(2*self._dim))
        X_lle = rbf_feature.fit_transform(X_sampled)

        # Create and train the XGBoost model with early stopping
        model_AddChis = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = rbf_feature.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_AddChis.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = rbf_feature.transform(self.train_x)
        # Predict probabilities for the training and testing sets
        train_probs = model_AddChis.predict_proba(train_data)
        cost_addChi = 1 - metrics.accuracy_score(self.y_val, model_AddChis.predict(val_data))
        print('cost_chi', cost_addChi)
        #-----------------------------------------------------------
        # step 7: goes to another round
        X_sampled, y_sampled = self._sampling(train_probs)

        feat_nystroem = Nystroem(gamma=.2,
                                        random_state=1,
                                        n_components=(int(self._dim*2)))
        X_lle = feat_nystroem.fit_transform(X_sampled)

        # Create and train the XGBoost model with early stopping
        model_nystroem = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = feat_nystroem.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_nystroem.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = feat_nystroem.transform(self.train_x)

        # Predict probabilities for the training and testing sets
        train_probs = model_nystroem.predict_proba(train_data)
        cost_nystroem = 1 - metrics.accuracy_score(self.y_val, model_nystroem.predict(val_data))
        print('cost_nystroem', cost_nystroem)


        # ----------------------------------------------------------
        # step 7: goes to another round
        X_sampled, y_sampled = self._sampling(train_probs)

        ps = PolynomialCountSketch(degree=3, random_state=1,n_components=(int(self._dim*2)))
        X_lle = ps.fit_transform(X_sampled)

        # Create and train the XGBoost model with early stopping
        model_ps = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10,  **config)
        val_data = ps.transform(self.val_x)
        eval_set = [ (val_data, self.y_val)]
        model_ps.fit(X_lle, y_sampled, eval_set=eval_set, verbose=False)
        # step 2.2 applied the data by dim Reduction method,
        train_data = ps.transform(self.train_x)

        # Predict probabilities for the training and testing sets
        train_probs = model_ps.predict_proba(train_data)
        cost_ps = 1 - metrics.accuracy_score(self.y_val, model_ps.predict(val_data))
        print('cost_nystroem', cost_ps)

        return cost_ps