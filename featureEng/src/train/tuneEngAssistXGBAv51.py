import os
import joblib
import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.train.utils import _build_datasets_UCI

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(123)


def get_score_(test_y, test_y_pred):
    f1_micro  = f1_score(test_y, test_y_pred, average="micro")
    f1_macro  = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc       = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred, average="macro", zero_division=0)
    recall    = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precision, recall


def load_data(path, splits='1'):
    return joblib.load(os.path.join(path, splits + '_data.pkl'))


class my_featExgboost:
    def __init__(self, params=None, output_path='', data_name=''):
        self.output_path = output_path
        self.data_name   = data_name
        self.param       = params

        train_x, train_y, test_x, test_y, val_x, val_y, space = _build_datasets_UCI(params)

        self.x,     self.test_x, self.val_x = train_x, test_x, val_x
        self.y,     self.test_y, self.val_y = train_y, test_y, val_y

        labels = np.unique(np.concatenate((train_y, test_y, val_y)))

        a = 0.4
        b = 0.3
        self.fact   = (1. - a) / b
        self.topN   = int(a * self.x.shape[0])
        self.randN  = int(b * self.x.shape[0])
        self.kClass = len(labels)

        self.param['multi_class'] = True if len(labels) > 2 else False
        self.n_boost   = 100
        self._n_trials = 100
        self._names    = ['None', 'auto', 'hpca', 'robuster', 'randP', 'minmax']
        self._bin      = min(max(20, self.x.shape[0] // (self.kClass ** 2)), 40)
        self.flag_feat = False

        # TỐI ƯU #3: early stopping
        self._patience = 10

        # ── Lưu thông tin lựa chọn model để xuất Excel ──
        self.model_choice      = None   # 'Feat-XGBoost' hoặc 'XGBoost'
        self.feat_acc_optuna   = 0.0    # acc của Feat-XGBoost trên val
        self.default_acc_optuna = 0.0   # acc của XGBoost gốc trên val

        self._load_data()

    # ─────────────────────────────────────────────────────────────
    def _load_data(self):
        self._data = {}
        for name in self._names:
            if name == 'None':
                self._data[name] = (self.x, self.test_x, self.val_x)
                continue
            data = load_data(
                os.path.join('.//Featdata', self.data_name),
                splits=(name + str(self.param['seed']))
            )
            if name == 'auto':
                if data[0] is None and not data[0]:
                    self._data[name] = (self.x, self.test_x, self.val_x)
                else:
                    self._data[name] = (data[0].values, data[1].values, data[2].values)
            else:
                if data[0] is None and not data[0]:
                    self._data[name] = (self.x, self.test_x, self.val_x)
                else:
                    self._data[name] = data

    # ─────────────────────────────────────────────────────────────
    def _sampling(self, Fm_, w_, _feat_type=''):
        data  = self._data[_feat_type]
        x_new = data[0]

        y_prev   = np.exp(Fm_) / np.exp(Fm_).sum(axis=1, keepdims=True)
        g        = -(np.eye(self.kClass)[self.y.squeeze()] - y_prev)
        sorted_g = np.argsort(np.linalg.norm(g, axis=1))[::-1]
        topSet   = sorted_g[:self.topN]
        randSet  = np.random.choice(sorted_g[self.topN:], size=self.randN, replace=False)
        usedSet  = np.hstack([topSet, randSet])
        w_[randSet] *= self.fact

        return x_new[usedSet], self.y[usedSet], w_[usedSet], Fm_[usedSet].ravel(), x_new

    # ─────────────────────────────────────────────────────────────
    def _train_model(self, configs):
        """TỐI ƯU #3: Early stopping boosting."""
        def base_model(x, y, weights, F0, config):
            model = XGBClassifier(
                n_estimators=1, base_score=None,
                objective='multi:softprob',
                num_class=self.kClass, **config
            )
            model.fit(x, y, sample_weight=weights, base_margin=F0, verbose=0)
            return model

        _Fm         = np.zeros((self.y.shape[0], self.kClass))
        _models     = []
        _best_acc   = -1.0
        _no_improve = 0

        for i in range(self.n_boost):
            _w = np.ones(shape=self.x.shape[0])
            input_x, input_y, w_, Fm_, x_old = self._sampling(
                _Fm, _w, _feat_type=self._names[i % 6]
            )
            newModel = base_model(input_x, input_y, w_, F0=Fm_, config=configs)
            _Fm     += newModel.predict(x_old, output_margin=True).reshape(-1, self.kClass)
            _models.append(newModel)

            if (i + 1) % 5 == 0:
                val_Fm   = np.zeros((self.val_x.shape[0], self.kClass))
                val_pred = self._predict(_models, val_Fm, _is_test=2)
                val_acc  = accuracy_score(self.val_y, val_pred)

                if val_acc > _best_acc + 1e-4:
                    _best_acc   = val_acc
                    _no_improve = 0
                else:
                    _no_improve += 1

                if _no_improve >= self._patience:
                    print(f'    Early stopping tại vòng {i+1}/{self.n_boost} '
                          f'(val_acc={_best_acc:.4f})')
                    break

        return _models

    # ─────────────────────────────────────────────────────────────
    def objective(self, trial):
        """TỐI ƯU #1: Optuna MedianPruner."""
        configs = {
            'eta':               trial.suggest_float('eta', 0.0001, 1.0, log=True),
            'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
            'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'max_depth':         trial.suggest_int('max_depth', 3, 20),
        }
        _models = self._train_model(configs)
        test_Fm = np.zeros((self.val_x.shape[0], self.kClass))
        y_pred  = self._predict(_models, test_Fm, _is_test=2)
        acc     = accuracy_score(y_true=self.val_y, y_pred=y_pred)

        trial.report(acc, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return acc

    # ─────────────────────────────────────────────────────────────
    def _default(self):
        def tune_xgboost(trial):
            params = {
                'eta':               trial.suggest_float('eta', 0.0001, 1.0, log=True),
                'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
                'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'max_depth':         trial.suggest_int('max_depth', 3, 20),
                'n_estimators':      trial.suggest_int('n_estimators', 8, 500),
                'objective':   'multi:softprob' if self.kClass > 2 else 'binary:logistic',
                'eval_metric': 'mlogloss'        if self.kClass > 2 else 'logloss',
            }
            bst = XGBClassifier(**params, early_stopping_rounds=10, verbosity=0)
            bst.fit(self.x, self.y, eval_set=[(self.val_x, self.val_y)], verbose=False)
            acc = accuracy_score(self.val_y, bst.predict(self.val_x))

            trial.report(acc, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return acc

        pruner    = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        study_xgb = optuna.create_study(direction='maximize', pruner=pruner)
        study_xgb.optimize(tune_xgboost, n_trials=self._n_trials)

        best_params = study_xgb.best_params
        best_params['objective']   = 'multi:softprob' if self.kClass > 2 else 'binary:logistic'
        best_params['eval_metric'] = 'mlogloss'        if self.kClass > 2 else 'logloss'
        return best_params, study_xgb.best_value

    # ─────────────────────────────────────────────────────────────
    def train_feat(self):
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        study  = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(self.objective, n_trials=self._n_trials)
        print(f'    [Feat-XGBoost] best_val_acc={study.best_value:.4f}  '
              f'trials={len(study.trials)}')
        return study.best_trial.params, study.best_value

    # ─────────────────────────────────────────────────────────────
    def train(self):
        """Chạy tuần tự, lưu thông tin lựa chọn model."""
        feat_params, feat_acc       = self.train_feat()
        default_params, default_acc = self._default()

        # Lưu lại để hyper_trainer có thể đọc
        self.feat_acc_optuna    = feat_acc
        self.default_acc_optuna = default_acc

        print(f'    [Feat-XGBoost] val_acc={feat_acc:.4f}')
        print(f'    [XGBoost gốc]  val_acc={default_acc:.4f}')

        if feat_acc > default_acc:
            self.flag_feat    = True
            self.model_choice = 'Feat-XGBoost'
            print('    → Chọn: Feat-XGBoost ✔')
            return feat_params
        else:
            self.flag_feat    = False
            self.model_choice = 'XGBoost'
            print('    → Chọn: XGBoost gốc ✔')
            return default_params

    # ─────────────────────────────────────────────────────────────
    def _predict(self, _models, test_Fm, _is_test=1):
        for id, model in enumerate(_models):
            input_x  = self._data[self._names[id % 6]][_is_test]
            test_Fm += model.predict(input_x, output_margin=True).reshape(-1, self.kClass)
        y_prob = np.exp(test_Fm) / np.exp(test_Fm).sum(axis=1, keepdims=True)
        return np.argmax(y_prob, axis=1)

    # ─────────────────────────────────────────────────────────────
    def predict(self, best_params):
        if not self.flag_feat:
            best_xgb_model = XGBClassifier(**best_params, early_stopping_rounds=10, verbosity=0)
            best_xgb_model.fit(self.x, self.y,
                               eval_set=[(self.val_x, self.val_y)], verbose=False)
            final_preds = best_xgb_model.predict(self.test_x)
        else:
            _models    = self._train_model(best_params)
            test_Fm    = np.zeros((self.test_x.shape[0], self.kClass))
            final_preds = self._predict(_models, test_Fm, _is_test=1)

        return get_score_(self.test_y, final_preds)