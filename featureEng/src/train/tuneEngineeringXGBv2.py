import os

import joblib
import optuna
import torch
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from sklearn.kernel_approximation import RBFSampler, Nystroem, PolynomialCountSketch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.train.utils import _build_datasets_UCI


def get_score_(test_y, test_y_pred):
    f1_micro = f1_score(test_y, test_y_pred, average="micro")
    f1_macro = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc = accuracy_score(test_y, test_y_pred)

    precesion = precision_score(test_y, test_y_pred, average="macro", zero_division=0)
    recall = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precesion, recall

def load_data(path, splits ='1'):
    return joblib.load(os.path.join(path, splits+'_data.pkl'))

class my_featExgboost:
    def __init__(self, params=None, output_path='', data_name= ''):
        self.output_path = output_path
        self.data_name = data_name
        self.param = params
        train_x, train_y, test_x, test_y, val_x, val_y, space = _build_datasets_UCI(params)

        self.x, self.test_x, self.val_x = train_x, test_x, val_x
        self.y, self.test_y, self.val_y = train_y, test_y, val_y

        labels = np.unique(np.concatenate((train_y, test_y, val_y)))

        a = 0.4  # sampling ratio of large gradient data
        b = 0.3  # sampling ratio of small gradient data
        self.fact = (1. - a) / b
        self.topN = int(a * self.x.shape[0])
        self.randN = int(b * self.x.shape[0])
        self.kClass = len(labels)
        # Initialize Fm with zeros for all classes (3 classes here)

        multi_class = True if len(labels) > 2 else False
        self.param['multi_class'] = multi_class
        self.n_boost = 100
        self._n_trials= 100
        self._names = ['None', 'auto', 'hpca', 'robuster',
                       'randP', 'minmax']
        self._load_data()
        self._bin = min(max(20, self.x.shape[0]//(self.kClass**2)), 40)


    def _load_data(self):
        # load feat data
        self._data ={}
        for name in self._names:
            if name == 'None':
                self._data[name] = (self.x, self.test_x, self.val_x)
                continue
            data = load_data(os.path.join('/home/fei/kon0278/python3/pycode/featureEng//Featdata//', self.data_name),
                             splits=(name + str(self.param['seed'])))
            if name == 'auto':
                self._data[name] = (data[0].values, data[1].values, data[2].values)
            else:
                self._data[name] = data

    def _sampling(self, Fm_, w_, _feat_type=''):
        # load feat data
        data = self._data[_feat_type]
        x_new, x_test_new, x_val_new = data[0], data[1], data[2]

        # return sample data with weights
        y_prev = np.exp(Fm_) / np.exp(Fm_).sum(axis=1, keepdims=True)  # softmax for probabilities
        g = -(np.eye(self.kClass)[self.y.squeeze()] - y_prev)  # first order gradients (one-hot encoding for y)

        # Sort by gradient magnitude (for the largest gradient sampling)
        sorted_g = np.argsort(np.linalg.norm(g, axis=1))[::-1]
        topSet = sorted_g[:self.topN]
        randSet = np.random.choice(sorted_g[self.topN:], size=self.randN, replace=False)
        usedSet = np.hstack([topSet, randSet])

        w_[randSet] *= self.fact  # Assign weight fact to the small gradient data

        return x_new[usedSet], self.y[usedSet], w_[usedSet], Fm_[usedSet].ravel(), x_new

    def _train_model(self, configs):
        def base_model(x, y, weights, F0, config):
            model = XGBClassifier(n_estimators=1,  # just 1 round
                                  max_bin=self._bin, tree_method='hist',
                                  base_score=None,
                                  objective='multi:softprob',  # multi-class classification
                                  num_class=self.kClass, **config)  # specify number of classes

            model.fit(x, y, sample_weight=weights, base_margin=F0)
            return model

        _Fm = np.zeros((self.y.shape[0], self.kClass))  # one logit per class

        _models = []
        for i in range(self.n_boost):
            # Train the model on the selected set
            _w = np.ones(shape=self.x.shape[0])  # initial sample weights.
            input_x, input_y, w_, Fm_, x_old = self._sampling(_Fm, _w, _feat_type= self._names[i%6])
            newModel = base_model(input_x, input_y, w_, F0=Fm_, config= configs)

            # Update Fm (logits) for the whole dataset
            _Fm += newModel.predict(x_old, output_margin=True).reshape(-1, self.kClass)
            _models.append(newModel)

        return _models

    def objective(self, trial):
        eta = trial.suggest_float('eta', 0.0001, 1.0, log=True)
        gamma = trial.suggest_float('gamma', 0.0, 1.0)
        subsample = trial.suggest_float('subsample', 0.4, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        colsample_bynode = trial.suggest_float('colsample_bynode', 0.5, 1.0)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        #max_bin = trial.suggest_int('max_bin', 20, 60)

        configs = {
            'eta': eta,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'max_depth': max_depth,
            #'max_bin': max_bin,
        }

        _models = self._train_model(configs)
        test_Fm = np.zeros((self.val_x.shape[0], self.kClass))
        y_pred = self._predict(_models, test_Fm, _is_test = 2)
        # Add predictions from all models (logits)
        acc = accuracy_score(y_true=self.val_y, y_pred=y_pred)

        return acc

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self._n_trials)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        print(study.best_value)
        return study.best_trial.params

    def _predict(self, _models, test_Fm, _is_test = 1):
        # Add predictions from all models (logits)
        for id, model in enumerate(_models):
            input_x = self._data[self._names[id%6]][_is_test]
            test_Fm += model.predict(input_x, output_margin=True).reshape(-1, self.kClass)

        # Convert logits to probabilities using softmax
        y_prob = np.exp(test_Fm) / np.exp(test_Fm).sum(axis=1, keepdims=True)
        # Class prediction is the argmax of the probabilities
        y_pred = np.argmax(y_prob, axis=1)

        return y_pred

    def predict(self, best_params):
        # You may need to adjust the keys if they have prefixes added by Optuna
        _models = self._train_model(best_params)
        test_Fm = np.zeros((self.test_x.shape[0], self.kClass))

        final_test_predictions = self._predict(_models, test_Fm, _is_test = 1)
        # Run the training process with the best parameters and evaluate on the test data
        acc, f1_macro, f1_micro, f1_weight, precision, recall = get_score_(self.test_y, final_test_predictions)

        # print('Test Accuracy:', acc)
        # print('F1 Macro:', f1_macro)
        # print('F1 Micro:', f1_micro)
        # print('F1 Weight:', f1_weight)
        # print('Precision:', precision)
        # print('Recall:', recall)
        return acc, f1_macro, f1_micro, f1_weight, precision, recall