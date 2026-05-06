"""
tuneEngAssistXGBAv51_bench.py

Các cải tiến đã tích hợp:
  1. Two-Stage Fine-tuning
  2. Class Weights
  3. DART Dropout
  4. Focal Loss
  5. Peak Threshold + Early Stopping

Fix từ Nhóm 2:
  FIX 1 - RAM O(n²)       : Dùng XGBoost enable_categorical, del sau fold
  FIX 2 - Feat selection   : Chỉ chọn Feat nếu val_acc > XGBoost + margin (0.005)
                             + SelectPercentile theo n_samples
  FIX 3 - Early Stop sớm  : MIN_ROUNDS=40, scale learning_rate theo n_samples
"""

import os
import gc
import warnings
import joblib
import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectPercentile, f_classif
from src.train.utils import _build_datasets_UCI

warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(123)

# ── FIX 3: Hằng số cho Early Stop ────────────────────────────────────────────
MIN_ROUNDS = 40   # Không dừng trước vòng này dù không cải thiện


# ── Focal Loss ────────────────────────────────────────────────────────────────
def focal_loss_gradient(y_pred, dtrain, gamma=2.0):
    y_true     = dtrain.get_label().astype(int)
    y_pred_exp = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
    p          = y_pred_exp / y_pred_exp.sum(axis=1, keepdims=True)
    one_hot    = np.zeros_like(p)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    p_t        = np.sum(p * one_hot, axis=1, keepdims=True)
    focal_w    = (1 - p_t) ** gamma
    grad = focal_w * (p - one_hot)
    hess = focal_w * p * (1 - p) * (
        1 + gamma * (1 - p_t) * np.log(p_t + 1e-8)
    )
    return grad, hess


def get_score_(test_y, test_y_pred):
    return (
        accuracy_score(test_y, test_y_pred),
        f1_score(test_y, test_y_pred, average="macro"),
        f1_score(test_y, test_y_pred, average="micro"),
        f1_score(test_y, test_y_pred, average="weighted"),
        precision_score(test_y, test_y_pred, average="macro", zero_division=0),
        recall_score(test_y, test_y_pred, average="macro", zero_division=0),
    )


def load_data(path, splits='1'):
    return joblib.load(os.path.join(path, splits + '_data.pkl'))


# ── FIX 2: Tính percentile feature selection theo n_samples ──────────────────
def _get_percentile(n_samples):
    """
    Điều chỉnh SelectPercentile theo kích thước dataset:
    - < 100 mẫu → giữ 90% features (dataset nhỏ, không nên bỏ nhiều)
    - < 300 mẫu → giữ 80% features
    - >= 300 mẫu → giữ 70% features
    """
    if n_samples < 100:
        return 90
    elif n_samples < 300:
        return 80
    else:
        return 70


# ── FIX 3: Scale learning rate theo n_samples ─────────────────────────────────
def _get_base_lr(n_samples):
    """
    Dataset nhỏ cần learning rate lớn hơn để hội tụ đủ nhanh.
    Dataset lớn cần learning rate nhỏ để ổn định.
    """
    if n_samples < 100:
        return 0.1
    elif n_samples < 300:
        return 0.05
    else:
        return 0.03


# ─────────────────────────────────────────────────────────────────────────────
class my_featExgboost:
    def __init__(self, params=None, output_path='', data_name='',
                 level_config=None, override_data=None):
        self.output_path = output_path
        self.data_name   = data_name
        self.param       = params

        cfg = level_config or {}

        # ── BẬT/TẮT từng cải tiến ───────────────────────────────────────
        self.use_pruning           = cfg.get('use_pruning',           False)
        self.use_early_stop        = cfg.get('use_early_stop',        False)
        self.use_dart              = cfg.get('use_dart',              False)
        self.use_class_weights     = cfg.get('use_class_weights',     False)
        self.use_focal_loss        = cfg.get('use_focal_loss',        False)
        self.use_two_stage         = cfg.get('use_two_stage',         False)
        self.use_weighted_ensemble = cfg.get('use_weighted_ensemble', False)
        self.use_feat_selection    = cfg.get('use_feat_selection',    False)
        self.use_min_rounds        = cfg.get('use_min_rounds',        True)
        self.use_scale_lr          = cfg.get('use_scale_lr',          False)
        self.use_feat_margin       = cfg.get('use_feat_margin',       False)
        self.use_ram_gc            = cfg.get('use_ram_gc',            True)
        self.use_peak_stop         = cfg.get('use_peak_stop',         False)

        # Focal Loss không tương thích với GOSS → tắt tự động
        if self.use_focal_loss:
            print('    ⚠ Focal Loss tắt tự động — không tương thích với GOSS')
            self.use_focal_loss = False

        # ── THAM SỐ SỐ HỌC ──────────────────────────────────────────────
        self._n_trials           = cfg.get('n_trials',          100)
        self._n_trials_fine      = cfg.get('n_trials_fine',     20)
        self.n_boost             = cfg.get('n_boost',           100)
        self._patience           = cfg.get('patience',          100)
        self._peak_threshold     = cfg.get('peak_threshold',    1.1)   # 1.1 = không bao giờ peak
        self._focal_gamma        = cfg.get('focal_gamma',       2.0)
        self._dart_rate_drop     = cfg.get('dart_rate_drop',    0.1)
        self._feat_margin        = cfg.get('feat_margin',       0.005)
        # Tính weight mỗi N vòng (1=mỗi vòng chậm, 10=mỗi 10 vòng nhanh hơn)
        self._weight_interval    = cfg.get('weight_interval',   10)

        # Load data
        if override_data is not None:
            train_x = np.array(override_data['train_x'])
            train_y = np.array(override_data['train_y']).ravel()
            test_x  = np.array(override_data['test_x'])
            test_y  = np.array(override_data['test_y']).ravel()
            val_x   = np.array(override_data['val_x'])
            val_y   = np.array(override_data['val_y']).ravel()
        else:
            train_x, train_y, test_x, test_y, val_x, val_y, _ = \
                _build_datasets_UCI(params)
            train_y = np.array(train_y).ravel()
            test_y  = np.array(test_y).ravel()
            val_y   = np.array(val_y).ravel()

        # Fix class label không liên tục
        all_labels = np.concatenate([train_y, test_y, val_y])
        self.le    = LabelEncoder()
        self.le.fit(all_labels)
        train_y = self.le.transform(train_y)
        test_y  = self.le.transform(test_y)
        val_y   = self.le.transform(val_y)

        self.x,     self.test_x, self.val_x = train_x, test_x, val_x
        self.y,     self.test_y, self.val_y = train_y, test_y, val_y

        labels      = np.unique(np.concatenate([train_y, test_y, val_y]))
        a, b        = 0.4, 0.3
        self.fact   = (1. - a) / b
        self.topN   = max(1, int(a * self.x.shape[0]))
        self.randN  = max(1, int(b * self.x.shape[0]))
        self.kClass = len(labels)

        self.param['multi_class'] = len(labels) > 2
        self._names  = ['None', 'auto', 'hpca', 'robuster', 'randP', 'minmax']
        self._bin    = min(max(20, self.x.shape[0] // (self.kClass ** 2)), 40)
        self.flag_feat = False

        # ── FIX 3: tính base learning rate theo n_samples ────────────────
        self._base_lr = _get_base_lr(len(train_x))

        # ── FIX 1: Class Weights ─────────────────────────────────────────
        if self.use_class_weights:
            self._sample_weights = compute_sample_weight(
                class_weight='balanced', y=self.y
            )
        else:
            self._sample_weights = np.ones(len(self.y))

        # Thông tin báo cáo
        self.model_choice        = None
        self.feat_acc_optuna     = 0.0
        self.default_acc_optuna  = 0.0
        self.early_stop_rounds   = []
        self.stop_reasons        = []

        # Weighted Ensemble: luu val_acc tung cay lam trong so
        self._model_weights = []

        # Winner tracking
        self.val_scores = {"Feat-XGB": 0.0, "Trad-XGB": 0.0}
        self.winner     = "Unknown"

        # Parallel Optuna trials (n_jobs=1 default vì Windows)
        self._n_jobs = cfg.get("n_jobs", 1)

        self._load_data()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_data(self):
        """
        FIX 1 (RAM): Kiểm tra cả số mẫu VÀ số features trước khi dùng Featdata.
        FIX 1 (RAM): Dùng gc.collect() sau khi load để giải phóng RAM.
        """
        self._data = {}
        for name in self._names:
            if name == 'None':
                self._data[name] = (self.x, self.test_x, self.val_x)
                continue
            try:
                data = load_data(
                    os.path.join('.//Featdata', self.data_name),
                    splits=(name + str(self.param['seed']))
                )
                if name == 'auto':
                    feat_x = data[0].values if data[0] is not None else None
                else:
                    feat_x = data[0] if data[0] is not None else None

                # FIX: kiểm tra cả số mẫu VÀ số features
                if (feat_x is not None
                        and len(feat_x) == len(self.x)
                        and feat_x.shape[1] == self.x.shape[1]):
                    if name == 'auto':
                        self._data[name] = (data[0].values,
                                            data[1].values,
                                            data[2].values)
                    else:
                        self._data[name] = data
                else:
                    self._data[name] = (self.x, self.test_x, self.val_x)

            except:
                self._data[name] = (self.x, self.test_x, self.val_x)

            # FIX 1 (RAM): giải phóng biến tạm
            gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_feature_selection(self, x_train, y_train, x_test, x_val):
        """
        FIX 2 (Feat selection): SelectPercentile theo n_samples.
        Giữ tối thiểu 70% features.
        """
        percentile = _get_percentile(len(x_train))
        try:
            selector = SelectPercentile(f_classif, percentile=percentile)
            x_train_sel = selector.fit_transform(x_train, y_train)
            x_test_sel  = selector.transform(x_test)
            x_val_sel   = selector.transform(x_val)
            return x_train_sel, x_test_sel, x_val_sel
        except:
            return x_train, x_test, x_val

    # ─────────────────────────────────────────────────────────────────────────
    def _sampling(self, Fm_, w_, _feat_type=''):
        x_new    = self._data[_feat_type][0]
        y_prev   = np.exp(Fm_) / np.exp(Fm_).sum(axis=1, keepdims=True)
        g        = -(np.eye(self.kClass)[self.y.squeeze()] - y_prev)
        sorted_g = np.argsort(np.linalg.norm(g, axis=1))[::-1]
        topSet   = sorted_g[:self.topN]
        randSet  = np.random.choice(sorted_g[self.topN:],
                                    size=self.randN, replace=False)
        usedSet  = np.hstack([topSet, randSet])
        w_[randSet] *= self.fact
        return (x_new[usedSet], self.y[usedSet],
                w_[usedSet], Fm_[usedSet].ravel(), x_new)

    # ─────────────────────────────────────────────────────────────────────────
    def _make_base_model(self, config):
        if self.use_dart:
            return XGBClassifier(
                n_estimators=1, base_score=None,
                booster='dart',
                rate_drop=self._dart_rate_drop,
                skip_drop=0.5,
                objective='multi:softprob',
                num_class=self.kClass,
                **config
            )
        return XGBClassifier(
            n_estimators=1, base_score=None,
            objective='multi:softprob',
            num_class=self.kClass,
            **config
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _train_model(self, configs):
        """
        FIX 3 (Early Stop): Không dừng trước MIN_ROUNDS.
        FIX 3 (LR): Scale learning rate theo n_samples.
        FIX 1 (RAM): del biến tạm và gc.collect() cuối vòng.
        """
        # Scale learning rate theo n_samples (chỉ khi bật)
        scaled_configs = dict(configs)
        if self.use_scale_lr and scaled_configs.get('eta', 1.0) < self._base_lr:
            scaled_configs['eta'] = self._base_lr

        _Fm         = np.zeros((self.y.shape[0], self.kClass))
        _models     = []
        self._model_weights = []   # Reset weights moi lan train
        _best_acc   = -1.0
        _no_improve = 0
        stopped_at  = self.n_boost
        stop_reason = 'full'

        for i in range(self.n_boost):
            _w = np.ones(shape=self.x.shape[0])
            input_x, input_y, w_, Fm_, x_old = self._sampling(
                _Fm, _w, _feat_type=self._names[i % 6]
            )

            # FIX 1 (RAM): tính class weight từ input_y thay vì index lookup
            if self.use_class_weights:
                class_w    = compute_sample_weight('balanced', y=input_y)
                combined_w = w_ * class_w
            else:
                combined_w = w_

            if self.use_focal_loss:
                import xgboost as xgb
                base_margin = Fm_ if len(Fm_) > 0 else None
                dtrain = xgb.DMatrix(
                    input_x, label=input_y,
                    weight=combined_w,
                    base_margin=base_margin
                )
                params_xgb = {
                    'max_depth':         scaled_configs.get('max_depth', 6),
                    'eta':               scaled_configs.get('eta', 0.1),
                    'gamma':             scaled_configs.get('gamma', 0),
                    'subsample':         scaled_configs.get('subsample', 0.8),
                    'colsample_bytree':  scaled_configs.get('colsample_bytree', 0.8),
                    'colsample_bylevel': scaled_configs.get('colsample_bylevel', 0.8),
                    'colsample_bynode':  scaled_configs.get('colsample_bynode', 0.8),
                    'num_class':         self.kClass,
                    'booster':           'dart' if self.use_dart else 'gbtree',
                }
                if self.use_dart:
                    params_xgb['rate_drop'] = self._dart_rate_drop
                    params_xgb['skip_drop'] = 0.5

                bst = xgb.train(
                    params_xgb, dtrain, num_boost_round=1,
                    obj=lambda p, d: focal_loss_gradient(
                        p.reshape(-1, self.kClass), d,
                        gamma=self._focal_gamma
                    ),
                )
                newModel    = bst
                dtest_old   = xgb.DMatrix(x_old)
                pred_margin = bst.predict(
                    dtest_old, output_margin=True
                ).reshape(-1, self.kClass)

                # FIX 1 (RAM): del biến tạm
                del dtrain, dtest_old
            else:
                newModel = self._make_base_model(scaled_configs)
                newModel.fit(input_x, input_y,
                             sample_weight=combined_w,
                             base_margin=Fm_, verbose=0)
                pred_margin = newModel.predict(
                    x_old, output_margin=True
                ).reshape(-1, self.kClass)

            _Fm += pred_margin
            _models.append(newModel)

            # ── Weighted Ensemble: tính weight từ CUMULATIVE ensemble ─────
            # Chỉ tính khi use_weighted_ensemble=True
            # Tính mỗi weight_interval vòng để tiết kiệm thời gian
            if self.use_weighted_ensemble:
                if (i + 1) % self._weight_interval == 0 or i == 0:
                    # Tính từ cumulative _Fm (đúng) không phải cây đơn lẻ (sai)
                    val_Fm_cum = np.zeros((self.val_x.shape[0], self.kClass))
                    for j, m in enumerate(_models):
                        vx = self._data[self._names[j % 6]][2]
                        if hasattr(m, 'predict') and hasattr(m, 'get_booster'):
                            val_Fm_cum += m.predict(
                                vx, output_margin=True
                            ).reshape(-1, self.kClass)
                        else:
                            import xgboost as xgb
                            val_Fm_cum += m.predict(
                                xgb.DMatrix(vx), output_margin=True
                            ).reshape(-1, self.kClass)
                    val_p = np.exp(val_Fm_cum) /                             np.exp(val_Fm_cum).sum(axis=1, keepdims=True)
                    w_i = accuracy_score(self.val_y, np.argmax(val_p, axis=1))
                    del val_Fm_cum, val_p
                else:
                    # Dùng lại weight của lần tính gần nhất
                    w_i = self._model_weights[-1] if self._model_weights else 1.0
                self._model_weights.append(w_i)
            else:
                # Tắt weighted ensemble → equal weight (cộng đều như gốc)
                self._model_weights.append(1.0)

            # Giải phóng RAM mỗi vòng (chỉ khi bật)
            if self.use_ram_gc:
                del input_x, input_y, w_, combined_w, pred_margin

            # Kiểm tra mỗi 5 vòng
            if self.use_early_stop and (i + 1) % 5 == 0:
                val_Fm   = np.zeros((self.val_x.shape[0], self.kClass))
                val_pred = self._predict(_models, val_Fm, _is_test=2)
                val_acc  = accuracy_score(self.val_y, val_pred)

                if val_acc > _best_acc + 1e-4:
                    _best_acc   = val_acc
                    _no_improve = 0
                else:
                    _no_improve += 1

                # Điều kiện 1: Chạm đỉnh (chỉ khi bật)
                if self.use_peak_stop and val_acc >= self._peak_threshold:
                    stopped_at  = i + 1
                    stop_reason = 'peak'
                    break

                # Điều kiện 2: Không cải thiện
                # use_min_rounds=True → không dừng trước MIN_ROUNDS
                min_r = MIN_ROUNDS if self.use_min_rounds else 0
                if (_no_improve >= self._patience
                        and (i + 1) >= min_r):
                    stopped_at  = i + 1
                    stop_reason = 'no_improve'
                    break

        # FIX 1 (RAM): giải phóng sau khi train xong
        gc.collect()

        self.early_stop_rounds.append(stopped_at)
        self.stop_reasons.append(stop_reason)
        return _models

    # ─────────────────────────────────────────────────────────────────────────
    def _make_study(self):
        if self.use_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=0)
            return optuna.create_study(direction='maximize', pruner=pruner)
        return optuna.create_study(direction='maximize')

    # ─────────────────────────────────────────────────────────────────────────
    def _make_configs(self, trial, narrow_around=None):
        if narrow_around is None:
            return {
                'eta':               trial.suggest_float('eta', 0.0001, 1.0, log=True),
                'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
                'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'max_depth':         trial.suggest_int('max_depth', 3, 20),
            }
        else:
            b = narrow_around
            def nf(name, lo, hi, log=False):
                center = b.get(name, (lo + hi) / 2)
                n_lo   = max(lo, center * 0.8)
                n_hi   = min(hi, center * 1.2)
                if n_lo >= n_hi: n_lo, n_hi = lo, hi
                return trial.suggest_float(name, n_lo, n_hi, log=log)
            return {
                'eta':               nf('eta', 0.0001, 1.0, log=True),
                'gamma':             nf('gamma', 0.0, 1.0),
                'subsample':         nf('subsample', 0.4, 1.0),
                'colsample_bytree':  nf('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': nf('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode':  nf('colsample_bynode', 0.5, 1.0),
                'max_depth':         trial.suggest_int(
                    'max_depth',
                    max(2, b.get('max_depth', 6) - 2),
                    min(20, b.get('max_depth', 6) + 2)
                ),
            }

    # ─────────────────────────────────────────────────────────────────────────
    def objective(self, trial, narrow_around=None):
        configs = self._make_configs(trial, narrow_around)
        _models = self._train_model(configs)
        test_Fm = np.zeros((self.val_x.shape[0], self.kClass))
        y_pred  = self._predict(_models, test_Fm, _is_test=2)
        acc     = accuracy_score(y_true=self.val_y, y_pred=y_pred)

        # FIX 1 (RAM): giải phóng sau mỗi trial
        del _models
        gc.collect()

        # Early stop khi dat perfect score
        if acc >= 1.0:
            trial.study.stop()

        if self.use_pruning:
            trial.report(acc, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return acc

    # ─────────────────────────────────────────────────────────────────────────
    def train_feat(self):
        """Two-Stage Fine-tuning với FIX 2: SelectPercentile trước."""
        # Feature selection (chỉ khi bật)
        if self.use_feat_selection:
            x_sel, x_test_sel, x_val_sel = self._apply_feature_selection(
                self.x, self.y, self.test_x, self.val_x
            )
        else:
            x_sel, x_test_sel, x_val_sel = self.x, self.test_x, self.val_x
        # Backup và swap data tạm thời
        x_orig, test_orig, val_orig = self.x, self.test_x, self.val_x
        self.x, self.test_x, self.val_x = x_sel, x_test_sel, x_val_sel
        # Cập nhật _data['None'] với features đã chọn
        self._data['None'] = (self.x, self.test_x, self.val_x)

        # Stage 1
        study1 = self._make_study()
        study1.optimize(
            lambda t: self.objective(t, narrow_around=None),
            n_trials=self._n_trials,
            n_jobs=self._n_jobs
        )
        best_coarse = study1.best_trial.params
        best_val1   = study1.best_value

        complete1 = sum(1 for t in study1.trials
                        if t.state == optuna.trial.TrialState.COMPLETE)
        pruned1   = sum(1 for t in study1.trials
                        if t.state == optuna.trial.TrialState.PRUNED)

        if self.use_two_stage:
            # Stage 2
            study2 = self._make_study()
            study2.optimize(
                lambda t: self.objective(t, narrow_around=best_coarse),
                n_trials=self._n_trials_fine
            )
            best_params = study2.best_trial.params
            best_val    = study2.best_value
            complete2   = sum(1 for t in study2.trials
                              if t.state == optuna.trial.TrialState.COMPLETE)
            pruned2     = sum(1 for t in study2.trials
                              if t.state == optuna.trial.TrialState.PRUNED)
            print(f'    [Feat] S1:{complete1}done/{pruned1}pruned '
                  f'S2:{complete2}done/{pruned2}pruned '
                  f'best={best_val:.4f} '
                  f'features={x_sel.shape[1]}/{x_orig.shape[1]}')
        else:
            best_params = best_coarse
            best_val    = best_val1
            print(f'    [Feat] {complete1}done/{pruned1}pruned '
                  f'best={best_val:.4f} '
                  f'features={x_sel.shape[1]}/{x_orig.shape[1]}')

        # Restore data gốc
        self.x, self.test_x, self.val_x = x_orig, test_orig, val_orig
        self._data['None'] = (self.x, self.test_x, self.val_x)

        # FIX 1 (RAM): giải phóng
        del x_sel, x_test_sel, x_val_sel
        gc.collect()

        return best_params, best_val

    # ─────────────────────────────────────────────────────────────────────────
    def _default(self):
        def tune_xgboost(trial, narrow_around=None):
            if narrow_around is None:
                p = {
                    'eta':               trial.suggest_float('eta', 0.0001, 1.0, log=True),
                    'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
                    'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                    'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
                    'max_depth':         trial.suggest_int('max_depth', 3, 20),
                    'n_estimators':      trial.suggest_int('n_estimators', 8, 500),
                }
            else:
                b = narrow_around
                p = {
                    'eta':               trial.suggest_float('eta',
                                             max(0.0001, b.get('eta', 0.1)*0.8),
                                             min(1.0, b.get('eta', 0.1)*1.2), log=True),
                    'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
                    'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                    'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
                    'max_depth':         trial.suggest_int('max_depth',
                                             max(2, b.get('max_depth', 6)-2),
                                             min(20, b.get('max_depth', 6)+2)),
                    'n_estimators':      trial.suggest_int('n_estimators', 8, 500),
                }

            p['objective']   = 'multi:softprob' if self.kClass > 2 else 'binary:logistic'
            p['eval_metric'] = 'mlogloss'        if self.kClass > 2 else 'logloss'

            bst = XGBClassifier(**p, early_stopping_rounds=10, verbosity=0)
            bst.fit(self.x, self.y,
                    sample_weight=self._sample_weights
                    if self.use_class_weights else None,
                    eval_set=[(self.val_x, self.val_y)],
                    verbose=False)
            acc = accuracy_score(self.val_y, bst.predict(self.val_x))

            # FIX 1 (RAM): giải phóng
            del bst
            gc.collect()

            if self.use_pruning:
                trial.report(acc, step=0)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return acc

        study1 = self._make_study()
        study1.optimize(
            lambda t: tune_xgboost(t, narrow_around=None),
            n_trials=self._n_trials
        )
        best_coarse = study1.best_params
        best_val1   = study1.best_value

        if self.use_two_stage:
            study2 = self._make_study()
            study2.optimize(
                lambda t: tune_xgboost(t, narrow_around=best_coarse),
                n_trials=self._n_trials_fine
            )
            best_params = study2.best_params
            best_val    = study2.best_value
        else:
            best_params = best_coarse
            best_val    = best_val1

        best_params['objective']   = 'multi:softprob' if self.kClass > 2 \
                                     else 'binary:logistic'
        best_params['eval_metric'] = 'mlogloss'        if self.kClass > 2 \
                                     else 'logloss'
        return best_params, best_val

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        feat_params, feat_acc       = self.train_feat()
        default_params, default_acc = self._default()

        self.feat_acc_optuna    = feat_acc
        self.default_acc_optuna = default_acc

        self.val_scores = {"Feat-XGB": feat_acc, "Trad-XGB": default_acc}

        # Perfect score → skip default và chọn Feat ngay
        if feat_acc >= 1.0:
            self.flag_feat    = True
            self.model_choice = 'Feat-XGBoost'
            self.winner       = 'Feat-XGBoost (Perfect Score)'
            print(f'    → Feat-XGBoost 🏆 Perfect Score!')
            return feat_params

        # Feat margin: chỉ chọn Feat nếu tốt hơn XGBoost + margin (khi bật)
        margin = self._feat_margin if self.use_feat_margin else 0.0
        if feat_acc > default_acc + margin:
            self.flag_feat    = True
            self.model_choice = 'Feat-XGBoost'
            self.winner       = 'Feat-XGBoost'
            print(f'    → Feat-XGBoost (feat={feat_acc:.4f} > '
                  f'xgb={default_acc:.4f} + margin={self._feat_margin})')
            return feat_params
        else:
            self.flag_feat    = False
            self.model_choice = 'XGBoost'
            self.winner       = 'Traditional-XGBoost'
            print(f'    → XGBoost (xgb={default_acc:.4f} >= '
                  f'feat={feat_acc:.4f} + margin={self._feat_margin})')
            return default_params

    # ─────────────────────────────────────────────────────────────────────────
    def _predict(self, _models, test_Fm, _is_test=1):
        """
        Weighted Ensemble Prediction:
        - Neu co _model_weights → moi cay nhan trong so = val_acc cua cay do
        - Cay chinh xac hon dong gop nhieu hon vao ket qua cuoi
        - Fallback ve equal weight neu chua co weights
        """
        import xgboost as xgb

        has_weights = (
            len(self._model_weights) == len(_models)
            and sum(self._model_weights) > 1e-8
        )

        for id, model in enumerate(_models):
            input_x = self._data[self._names[id % 6]][_is_test]
            if hasattr(model, 'predict') and hasattr(model, 'get_booster'):
                logits = model.predict(
                    input_x, output_margin=True
                ).reshape(-1, self.kClass)
            else:
                logits = model.predict(
                    xgb.DMatrix(input_x), output_margin=True
                ).reshape(-1, self.kClass)

            if has_weights:
                # Nhan trong so = val_acc cua cay nay
                test_Fm += self._model_weights[id] * logits
            else:
                # Fallback: equal weight nhu goc
                test_Fm += logits

        y_prob = np.exp(test_Fm) / np.exp(test_Fm).sum(axis=1, keepdims=True)
        return np.argmax(y_prob, axis=1)

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, best_params):
        if not self.flag_feat:
            p = {k: v for k, v in best_params.items()
                 if k not in ['objective', 'eval_metric']}
            p['objective']   = best_params.get('objective',
                'multi:softprob' if self.kClass > 2 else 'binary:logistic')
            p['eval_metric'] = best_params.get('eval_metric',
                'mlogloss' if self.kClass > 2 else 'logloss')
            bst = XGBClassifier(**p, early_stopping_rounds=10, verbosity=0)
            bst.fit(self.x, self.y,
                    sample_weight=self._sample_weights
                    if self.use_class_weights else None,
                    eval_set=[(self.val_x, self.val_y)],
                    verbose=False)
            final_preds = bst.predict(self.test_x)

            # FIX 1 (RAM): giải phóng
            del bst
            gc.collect()
        else:
            _models     = self._train_model(best_params)
            test_Fm     = np.zeros((self.test_x.shape[0], self.kClass))
            final_preds = self._predict(_models, test_Fm, _is_test=1)
            del _models
            gc.collect()

        if self.stop_reasons:
            peak_cnt = self.stop_reasons.count('peak')
            noi_cnt  = self.stop_reasons.count('no_improve')
            full_cnt = self.stop_reasons.count('full')
            print(f'    Stop: 🏆peak={peak_cnt} '
                  f'⏹no_improve={noi_cnt} ✅full={full_cnt}')

        return get_score_(self.test_y, final_preds)