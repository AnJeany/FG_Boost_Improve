import optuna
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

class my_tune:
    def __init__(self, train_x=None, val_x=None, test_x=None, num_class=0, train_y=None, val_y=None, test_y=None):
        self.train_x = torch.tensor(train_x) if not isinstance(train_x, torch.Tensor) else train_x
        self.val_x = torch.tensor(val_x) if not isinstance(val_x, torch.Tensor) else val_x
        self.test_x = torch.tensor(test_x) if not isinstance(test_x, torch.Tensor) else test_x
        self.kClasses = num_class
        self.train_y = torch.tensor(train_y) if not isinstance(train_y, torch.Tensor) else train_y
        self.y_val = torch.tensor(val_y) if not isinstance(val_y, torch.Tensor) else val_y
        self.y_test = torch.tensor(test_y) if not isinstance(test_y, torch.Tensor) else test_y

        # self.train_x = train_x
        # self.val_x = val_x
        # self.test_x = test_x
        # self.kClasses = num_class
        # self.train_y = train_y
        # self.y_val = val_y
        # self.y_test = test_y
        self.train_y = self.train_y.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()

        self._len = self.train_x.shape[0]
        self._dim = self.train_x.shape[1]

    def get_random_forest(self):
        # define the optimize function
        def tune_random_forest(trial):
            n_estimators = trial.suggest_int('n_estimators', 5, 100)
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 6)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 40)
            max_depth = trial.suggest_int('max_depth', 5, 40)

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize RandomForest
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(tune_random_forest, n_trials=100)
        best_rf_params = study_rf.best_params

        # Train final models with best parameters
        best_rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
        best_rf_model.fit(self.train_x, self.train_y)
        preds_ = best_rf_model.predict(self.test_x)
        return preds_

    def get_svm(self):
        # define the optimize function
        def tune_svm(trial):
            # kernel = trial.suggest_categorical('kernel', ['linear', ])#'poly', 'rbf', 'sigmoid'
            C = trial.suggest_float('C', 0.001, 1000.0, log=True)
            shrinking = trial.suggest_categorical('shrinking', [True, False])
            degree = trial.suggest_int('degree', 1, 5)
            coef0 = trial.suggest_float('coef0', 0.0, 10.0)
            # gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])

            clf = SVC(
                # kernel=kernel,
                C=C,
                shrinking=shrinking,
                degree=degree,
                coef0=coef0,
                # gamma=gamma,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            # print(accuracy)
            return accuracy

        # Optimize SVM
        study_svm = optuna.create_study(direction='maximize')
        study_svm.optimize(tune_svm, n_trials=100)
        best_svm_params = study_svm.best_params

        best_svm_model = SVC(**best_svm_params, random_state=42)
        best_svm_model.fit(self.train_x, self.train_y)
        preds_ = best_svm_model.predict(self.test_x)
        return preds_

    def get_gaussian_nb(self):
        def tune_gaussian_nb(trial):
            var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1e-2, log=True)

            clf = GaussianNB(var_smoothing=var_smoothing)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize GaussianNB
        study_gnb = optuna.create_study(direction='maximize')
        study_gnb.optimize(tune_gaussian_nb, n_trials=100)
        best_gnb_params = study_gnb.best_params

        # Train final models with best parameters
        best_model = GaussianNB(**best_gnb_params)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_knn(self):
        def tune_knn(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 2)

            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize KNeighborsClassifier
        study_knn = optuna.create_study(direction='maximize')
        study_knn.optimize(tune_knn, n_trials=100)
        best_knn_params = study_knn.best_params

        best_model = KNeighborsClassifier(**best_knn_params)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_gaussian_process(self):
        def tune_gaussian_process(trial):
            kernel = RBF(length_scale=trial.suggest_float('length_scale', 1e-3, 1e5, log=True))
            clf = GaussianProcessClassifier(kernel=kernel, random_state=42)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize GaussianProcessClassifier
        study_gpc = optuna.create_study(direction='maximize')
        study_gpc.optimize(tune_gaussian_process, n_trials=100)
        best_gpc_params = study_gpc.best_params
        best_model = GaussianProcessClassifier(kernel=RBF(length_scale=best_gpc_params['length_scale']),
                                                   random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_adaboost(self):
        def tune_adaboost(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)

            clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,  algorithm='SAMME',random_state=42)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize AdaBoostClassifier
        study_ada = optuna.create_study(direction='maximize')
        study_ada.optimize(tune_adaboost, n_trials=100)
        best_ada_params = study_ada.best_params
        best_model = AdaBoostClassifier(**best_ada_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_qda(self):
        def tune_qda(trial):
            reg_param = trial.suggest_float('reg_param', 0.0, 1.0)

            clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize QuadraticDiscriminantAnalysis
        study_qda = optuna.create_study(direction='maximize')
        study_qda.optimize(tune_qda, n_trials=100)
        best_qda_params = study_qda.best_params
        best_model = QuadraticDiscriminantAnalysis(**best_qda_params)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_decision_tree(self):
        def tune_decision_tree(trial):
            max_depth = trial.suggest_int('max_depth', 1, 32)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)

            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, random_state=42)
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize DecisionTreeClassifier
        study_dt = optuna.create_study(direction='maximize')
        study_dt.optimize(tune_decision_tree, n_trials=100)
        best_dt_params = study_dt.best_params
        best_model = DecisionTreeClassifier(**best_dt_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_hist_gradient_boosting(self):
        def tune_hist_gradient_boosting(trial):
            max_iter = trial.suggest_int('max_iter', 100, 1000)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            l2_regularization = trial.suggest_float('l2_regularization', 1e-10, 1.0, log=True)

            clf = HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                l2_regularization=l2_regularization,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize HistGradientBoostingClassifier
        study_hgb = optuna.create_study(direction='maximize')
        study_hgb.optimize(tune_hist_gradient_boosting, n_trials=100)
        best_hgb_params = study_hgb.best_params

        # Train final models with best parameters
        best_model = HistGradientBoostingClassifier(**best_hgb_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_bagging(self):
        def tune_bagging(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
            max_features = trial.suggest_float('max_features', 0.1, 1.0)
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])

            clf = BaggingClassifier(
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize BaggingClassifier
        study_bagging = optuna.create_study(direction='maximize')
        study_bagging.optimize(tune_bagging, n_trials=100)
        best_bagging_params = study_bagging.best_params

        best_model = BaggingClassifier(**best_bagging_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_extra_trees(self):
        def tune_extra_trees(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

            clf = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize ExtraTreesClassifier
        study_et = optuna.create_study(direction='maximize')
        study_et.optimize(tune_extra_trees, n_trials=100)
        best_et_params = study_et.best_params

        best_model = ExtraTreesClassifier(**best_et_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_gradient_boosting(self):
        def tune_gradient_boosting(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)

            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize GradientBoostingClassifier
        study_gb = optuna.create_study(direction='maximize')
        study_gb.optimize(tune_gradient_boosting, n_trials=100)
        best_gb_params = study_gb.best_params

        best_model = GradientBoostingClassifier(**best_gb_params, random_state=42)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_stacking(self):
        # Dictionary to map string names to actual estimators
        final_estimators_dict = {
            'lr': LogisticRegression(),
            'dt': DecisionTreeClassifier(),
            'svc': SVC(probability=True)
        }
        def tune_stacking(trial):
            estimators = [
                ('lr', LogisticRegression()),
                ('dt', DecisionTreeClassifier()),
                ('svc', SVC(probability=True))
            ]

            final_estimator_name = trial.suggest_categorical('final_estimator', ['lr', 'dt', 'svc'])
            final_estimator = final_estimators_dict[final_estimator_name]

            clf = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                passthrough=True,
                cv=5
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize StackingClassifier
        study_stacking = optuna.create_study(direction='maximize')
        study_stacking.optimize(tune_stacking, n_trials=100)
        best_stacking_params = study_stacking.best_params

        estimators_stacking = [
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC(probability=True))
        ]
        # Train final StackingClassifier with best parameters
        estimators_stacking = [
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC(probability=True))
        ]
        best_final_estimator = final_estimators_dict[best_stacking_params['final_estimator']]
        best_model = StackingClassifier(estimators=estimators_stacking, final_estimator=best_final_estimator,
                                                 passthrough=True, cv=5)
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_

    def get_voting(self):
        def tune_voting(trial):
            estimators = [
                ('lr', LogisticRegression()),
                ('dt', DecisionTreeClassifier()),
                ('svc', SVC(probability=True))
            ]
            voting = trial.suggest_categorical('voting', ['hard', 'soft'])

            clf = VotingClassifier(
                estimators=estimators,
                voting=voting
            )
            clf.fit(self.train_x, self.train_y)
            preds = clf.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize VotingClassifier
        study_voting = optuna.create_study(direction='maximize')
        study_voting.optimize(tune_voting, n_trials=100)
        best_voting_params = study_voting.best_params


        estimators_voting = [
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC(probability=True))
        ]
        best_model = VotingClassifier(estimators=estimators_voting, voting=best_voting_params['voting'])
        best_model.fit(self.train_x, self.train_y)
        preds_ = best_model.predict(self.test_x)
        return preds_


    def get_xgboost(self):
        # define the optimize function
        def tune_xgboost(trial):
            eta = trial.suggest_float('eta', 0.0001, 1.0, log=True)
            gamma = trial.suggest_float('gamma', 0.0, 1.0)
            subsample = trial.suggest_float('subsample', 0.4, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
            colsample_bynode = trial.suggest_float('colsample_bynode', 0.5, 1.0)
            max_depth = trial.suggest_int('max_depth', 3, 40)
            n_estimators = trial.suggest_int('n_estimators', 8, 500)

            params = {
                'eta': eta,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'colsample_bylevel': colsample_bylevel,
                'colsample_bynode': colsample_bynode,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'objective': 'multi:softprob' if self.kClasses > 2 else 'binary:logistic',
                'eval_metric': 'mlogloss' if self.kClasses > 2 else 'logloss'
            }

            bst = xgb.XGBClassifier(**params, early_stopping_rounds=10, verbosity=0)
            bst.fit(self.train_x, self.train_y, eval_set=[(self.val_x, self.y_val)],
                    verbose=False)
            preds = bst.predict(self.val_x)
            accuracy = accuracy_score(self.y_val, preds)
            return accuracy

        # Optimize XGBoost
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(tune_xgboost, n_trials=100)

        # Best model parameters
        best_xgb_params = study_xgb.best_params
        best_xgb_params['objective'] = 'multi:softprob' if self.kClasses > 2 else 'binary:logistic'
        best_xgb_params['eval_metric'] = 'mlogloss' if self.kClasses > 2 else 'logloss'

        # Train the best model with early stopping
        best_xgb_model = xgb.XGBClassifier(**best_xgb_params, early_stopping_rounds=10, verbosity=0)
        best_xgb_model.fit(self.train_x, self.train_y, eval_set=[(self.val_x, self.y_val)],
                           verbose=False)

        # Predict using the best model
        preds_ = best_xgb_model.predict(self.test_x)
        return preds_

    def print(self):
        print(f"RandomForest Accuracy: {0}")
        print(f"SVM Accuracy: {1}")
        print(f"XGBoost Accuracy: {2}")
        print(f"GaussianNB Accuracy: {3}")
        print(f"KNeighborsClassifier Accuracy: {4}")
        print(f"GaussianProcessClassifier Accuracy: {5}")
        print(f"QuadraticDiscriminantAnalysis Accuracy: {6}")
        print(f"AdaBoostClassifier Accuracy: {7}")
        print(f"DecisionTreeClassifier Accuracy: {8}")
        print(f"HistGradientBoostingClassifier Accuracy: {9}")
        print(f"BaggingClassifier Accuracy: {10}")
        print(f"ExtraTreesClassifier Accuracy: {11}")
        print(f"GradientBoostingClassifier Accuracy: {12}")
        print(f"StackingClassifier Accuracy: {13}")
        print(f"VotingClassifier Accuracy: {14}")
