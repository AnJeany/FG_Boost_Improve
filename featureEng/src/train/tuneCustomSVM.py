import optuna
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from sklearn.kernel_approximation import RBFSampler, Nystroem, PolynomialCountSketch
from src.dimRedu.SupervisedLLE import SupervisedLocallyLinearEmbedding
from src.dimRedu.supervisedISOMAP import SupervisedIsomap
from src.dimRedu.supervisedMds import FastSupervisedMDS
from src.dimRedu.supervisedSpectrumCluster import SupervisedSpectralEmbedding
from src.dimRedu.supervised_fast_MVU import SupervisedFastMVU
from sklearn.svm import SVC

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
def get_score_(test_y, test_y_pred):
    f1_micro = f1_score(test_y, test_y_pred, average="micro")
    f1_macro = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc = accuracy_score(test_y, test_y_pred)

    precesion = precision_score(test_y, test_y_pred, average="macro", zero_division=0)
    recall = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precesion, recall

class my_tuneSVM:
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
        self.train_probs = torch.rand(self._len, self.kClasses)
        self._n_trials= 20
        self._names = ['SupervisedLocallyLinearEmbedding', 'SupervisedFastMVU', 'SupervisedMDS',
                       'SupervisedSpectralEmbedding', 'SupervisedIsomap', 'RBFSampler', 'Nystroem',
                       'PolynomialCountSketch']
        _, _counts = np.unique(self.train_y, return_counts=True)

        self.n_nei_min = min(max(8, self._len // 32), self._len // 8)
        self.n_nei_max = max(self._len // 8, self._len // 4)

        self.n_marks_min = int(0.1 * self._len)
        self.n_marks_max = int(0.3 * self._len)
        self._ratio = 0.80
        # self.max_comp = int(self._len*self._ratio)
        self.n_up_min = int(min(self._len*0.8-1, self._dim * 1.5))
        # self.n_up_max = int(min(self._len, self._dim * 3))
        self.n_up_max = int(min(self._len*0.8, self._dim * 3))


    def _sampling(self, train_probs):
        ratio = self._ratio
        if not isinstance(train_probs, torch.Tensor):
            train_probs = torch.tensor(train_probs)

        train_probs = train_probs / train_probs.sum(dim=1, keepdim=True)
        train_uncertainty = torch.var(train_probs, dim=1)
        sampling_probabilities = train_uncertainty / train_uncertainty.sum()

        num_samples = min(int(self._len * ratio), self._len - self.kClasses)
        X_train_sampled = []
        y_train_sampled = []

        sss = StratifiedShuffleSplit(n_splits=1, train_size=num_samples, random_state=42)
        for train_index, _ in sss.split(self.train_x.numpy(), self.train_y.numpy().ravel()):
            unique_classes = self.train_y.unique()
            remaining_train_indices = list(train_index)

            # Ensure each class with fewer than 10 instances is fully included
            for cls in unique_classes:
                cls_indices = train_index[self.train_y[train_index].ravel() == cls]
                if len(cls_indices) < 10:
                    X_train_sampled.append(self.train_x[cls_indices])
                    y_train_sampled.append(self.train_y[cls_indices])
                    remaining_train_indices = [idx for idx in remaining_train_indices if idx not in cls_indices]

            # Adjust num_samples to account for the directly included instances
            len_ = len(torch.cat(y_train_sampled) if y_train_sampled else torch.tensor([], dtype=torch.long))
            num_samples -= len_

            remaining_train_indices = np.array(remaining_train_indices)
            for cls in unique_classes:
                cls_indices = remaining_train_indices[self.train_y[remaining_train_indices].ravel() == cls]
                if len(cls_indices) > 0:
                    cls_probabilities = sampling_probabilities[cls_indices]
                    cls_probabilities /= cls_probabilities.sum()

                    num_cls_samples = int(num_samples * (len(cls_indices) / len(remaining_train_indices)))
                    sampled_cls_indices = np.random.choice(cls_indices, size=num_cls_samples, replace=True,
                                                           p=cls_probabilities.numpy())

                    X_train_sampled.append(self.train_x[sampled_cls_indices])
                    y_train_sampled.append(self.train_y[sampled_cls_indices])

        return torch.cat(X_train_sampled), torch.cat(y_train_sampled)

    def _train_model(self, X_sampled, y_sampled, val_x, config, dim_reduction_method):
        clf = SVC(
            # kernel=kernel,
            C=config['C'],
            shrinking=config['shrinking'],
            degree=config['degree'],
            coef0=config['coef0'],
            random_state=42,
            probability = True,
        )
        if dim_reduction_method is not None:
            X_sampled = dim_reduction_method.fit_transform(X_sampled, y_sampled.ravel())

        clf.fit(X_sampled, y_sampled)

        return clf

    def objective(self, trial):
        config = {}
        for name in self._names:
            config[name] = {
            'C' : trial.suggest_float((name+'C'), 0.001, 1000.0, log=True),
            'shrinking' :  trial.suggest_categorical((name+'shrinking'), [True, False]),
            'degree':  trial.suggest_int((name+'degree'), 1, 5),
            'coef0':  trial.suggest_float((name+'coef0'), 0.0, 10.0)
        }

        dim_reduction_params = {}

        dim_reduction_params['SupervisedLocallyLinearEmbedding'] = {
            'n_neighbors': trial.suggest_int('lle_n_neighbors', self.n_nei_min, self.n_nei_max),
            'n_components': trial.suggest_int('lle_n_components', self._dim // 4, self._dim // 1.5)
        }
        dim_reduction_params['SupervisedFastMVU'] = {
            'n_components': trial.suggest_int('mvu_n_components', self._dim // 4, self._dim // 1.5),
            'n_landmarks': trial.suggest_int('mvu_n_landmarks', self.n_marks_min, self.n_marks_max)
        }
        dim_reduction_params['SupervisedMDS'] = {
            'n_components': trial.suggest_int('mds_n_components', self._dim // 4, self._dim // 1.5),
            'n_landmarks': trial.suggest_int('mds_n_landmarks', self.n_marks_min, self.n_marks_max)
        }
        # dim_reduction_params['UMAP'] = {
        #     'n_neighbors': trial.suggest_int('umap_n_neighbors', self._len// max(self.kClasses, 20), self._len// min(self.kClasses, 20)),
        #     'n_components': trial.suggest_int('umap_n_components', self._dim // 4, self._dim // 2)
        # }
        dim_reduction_params['SupervisedSpectralEmbedding'] = {
            'n_components': trial.suggest_int('se_n_components', self._dim // 4, self._dim // 1.5),
            'n_neighbors': trial.suggest_int('se_n_neighbors', self.n_nei_min, self.n_nei_max),
        }
        dim_reduction_params['SupervisedIsomap'] = {
            'n_neighbors': trial.suggest_int('isomap_n_neighbors', (self.n_nei_min*2), self.n_nei_max*4),
            'n_components': trial.suggest_int('isomap_n_components', self._dim // 4, self._dim // 1.5)
        }
        dim_reduction_params['RBFSampler'] = {
            'gamma': trial.suggest_float('rbf_gamma', 0.01, 1.0, log=True),
            'n_components': trial.suggest_int('rbf_n_components', self.n_up_min, self.n_up_max)
        }
        dim_reduction_params['Nystroem'] = {
            'gamma': trial.suggest_float('nystroem_gamma', 0.01, 1.0, log=True),
            'n_components': trial.suggest_int('nystroem_n_components',  self.n_up_min, self.n_up_max)
        }
        dim_reduction_params['PolynomialCountSketch'] = {
            'degree': trial.suggest_int('pcs_degree', 2, 5),
            'n_components': trial.suggest_int('pcs_n_components',  self.n_up_min, self.n_up_max)
        }


        dim_reduction_methods = [
            SupervisedLocallyLinearEmbedding(**dim_reduction_params['SupervisedLocallyLinearEmbedding']),
            SupervisedFastMVU(**dim_reduction_params['SupervisedFastMVU']),
            # umap.UMAP(**dim_reduction_params['UMAP']),
            FastSupervisedMDS(**dim_reduction_params['SupervisedMDS']),
            SupervisedSpectralEmbedding(**dim_reduction_params['SupervisedSpectralEmbedding']),
            SupervisedIsomap(**dim_reduction_params['SupervisedIsomap']),
            RBFSampler(**dim_reduction_params['RBFSampler']),
            Nystroem(**dim_reduction_params['Nystroem']),
            PolynomialCountSketch(**dim_reduction_params['PolynomialCountSketch']),
            # SupervisedLDA(),
        ]

        costs = []
        train_probs = self.train_probs
        val_probs_list,test_probs_list = [], []
        for id, dim_reduction_method in enumerate(dim_reduction_methods): #have to keep the
            try:
                X_sampled, y_sampled = self._sampling(train_probs)
                model = self._train_model(X_sampled, y_sampled.ravel(), self.val_x, config[self._names[id]], dim_reduction_method)
                val_data = dim_reduction_method.transform(self.val_x)
                train_data = dim_reduction_method.transform(self.train_x)
                train_probs = model.predict_proba(train_data)
                cost = 1 - metrics.accuracy_score(self.y_val, model.predict(val_data))
                val_probs = model.predict_proba(val_data)
                val_probs_list.append(val_probs)
                costs.append(cost)

            except Exception as e:
                costs.append(1.0)

        # Calculate weights inversely proportional to the costs --- based on weighted results
        weights = np.array(costs)
        weights = 1 / (weights + 1e-8)  # Avoid division by zero
        weights /= weights.sum()  # Normalize to sum to 1
        # Weighted average of test probabilities
        final_val_probs = sum(w * tp for w, tp in zip(weights, val_probs_list))
        # Final prediction and accuracy
        final_val_predictions = np.argmax(final_val_probs, axis=1)
        final_accuracy = metrics.accuracy_score(self.y_val, final_val_predictions)

        return final_accuracy

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self._n_trials)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        print(study.best_value)
        return study.best_trial.params

    def _predict(self, dim_reduction_params, config):
        dim_reduction_methods = [
            SupervisedLocallyLinearEmbedding(**dim_reduction_params['SupervisedLocallyLinearEmbedding']),
            SupervisedFastMVU(**dim_reduction_params['SupervisedFastMVU']),
            ## umap.UMAP(**dim_reduction_params['UMAP']),
            FastSupervisedMDS(**dim_reduction_params['SupervisedMDS']),
            SupervisedSpectralEmbedding(**dim_reduction_params['SupervisedSpectralEmbedding']),
            SupervisedIsomap(**dim_reduction_params['SupervisedIsomap']),
            RBFSampler(**dim_reduction_params['RBFSampler']),
            Nystroem(**dim_reduction_params['Nystroem']),
            PolynomialCountSketch(**dim_reduction_params['PolynomialCountSketch']),
            ## SupervisedLDA(),
        ]

        costs = []
        train_probs = self.train_probs
        test_probs_list = []
        for id, dim_reduction_method in enumerate(dim_reduction_methods):
            try:
                X_sampled, y_sampled = self._sampling(train_probs)
                model = self._train_model(X_sampled, y_sampled.ravel(), self.val_x, config[self._names[id]], dim_reduction_method)
                val_data = dim_reduction_method.transform(self.val_x)
                train_data = dim_reduction_method.transform(self.train_x)
                train_probs = model.predict_proba(train_data)
                cost = 1 - metrics.accuracy_score(self.y_val, model.predict(val_data))
                costs.append(cost)
                test_data = dim_reduction_method.transform(self.test_x)
                test_probs = model.predict_proba(test_data)
                test_probs_list.append(test_probs)

            except Exception as e:
                costs.append(1.0)

        # Calculate weights inversely proportional to the costs --- based on weighted results
        weights = np.array(costs)
        weights = 1 / (weights + 1e-8)  # Avoid division by zero
        weights /= weights.sum()  # Normalize to sum to 1
        # Weighted average of test probabilities
        final_test_probs = sum(w * tp for w, tp in zip(weights, test_probs_list))
        # Final prediction and accuracy
        final_test_predictions = np.argmax(final_test_probs, axis=1)
        return final_test_predictions

    def predict(self, best_params):
        # You may need to adjust the keys if they have prefixes added by Optuna
        svm_params = {
        }

        for name in self._names:
            svm_params[name] = {
            'C': best_params[name+'C'],
            'shrinking': best_params[name+'shrinking'],
            'degree': best_params[name+'degree'],
            'coef0': best_params[name+'coef0'],
        }
        # Dimensionality reduction method parameters
        dim_reduction_params = {
            'SupervisedLocallyLinearEmbedding': {
                'n_neighbors': best_params['lle_n_neighbors'],
                'n_components': best_params['lle_n_components']
            },
            'SupervisedFastMVU': {
                'n_components': best_params['mvu_n_components'],
                'n_landmarks': best_params['mvu_n_landmarks']
            },
            'SupervisedMDS': {
                'n_components': best_params['mds_n_components'],
                'n_landmarks': best_params['mds_n_landmarks']
            },
            'SupervisedSpectralEmbedding': {
                'n_components': best_params['se_n_components'],
                'n_neighbors': best_params['se_n_neighbors']
            },
            'SupervisedIsomap': {
                'n_neighbors': best_params['isomap_n_neighbors'],
                'n_components': best_params['isomap_n_components']
            },
            'RBFSampler': {
                'gamma': best_params['rbf_gamma'],
                'n_components': best_params['rbf_n_components']
            },
            'Nystroem': {
                'gamma': best_params['nystroem_gamma'],
                'n_components': best_params['nystroem_n_components']
            },
            'PolynomialCountSketch': {
                'degree': best_params['pcs_degree'],
                'n_components': best_params['pcs_n_components']
            },
            # 'SupervisedLDA': {}
        }

        final_test_predictions = self._predict(dim_reduction_params, svm_params)
        # Run the training process with the best parameters and evaluate on the test data
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score_(self.y_test, final_test_predictions)

        # print('Test Accuracy:', acc)
        # print('F1 Macro:', f1_macro)
        # print('F1 Micro:', f1_micro)
        # print('F1 Weight:', f1_weight)
        # print('Precision:', precision)
        # print('Recall:', recall)
        return acc, f1_macro, f1_micro, f1_weight, precision, recall