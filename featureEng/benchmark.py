"""
benchmark.py
Chạy benchmark với:
  - 10 cấp dữ liệu: 10%, 20%, ..., 100%
  - 2 phương án: Baseline và Full Optimization (M10)
  - Tự động tìm dataset có sẵn
  - Fix lỗi data ít ở cấp thấp

Cách dùng:
    python featureEng/benchmark.py
"""

import os
import sys
import time
import tracemalloc
import yaml
import numpy as np
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'train'))

from src.train.utils import set_seeds, expanduservars
from benchmark_excel import save_benchmark_to_excel
from src.train.tuneEngAssistXGBAv51_bench import my_featExgboost

print("=== benchmark.py đang chạy ===")

# ─── 2 phương án ─────────────────────────────────────────────────────────────

METHODS = {
    # ── Baseline: Không có cải tiến nào ─────────────────────────────────────
    'baseline': {
        'name':        'Baseline',
        'description': 'Chạy như gốc — không có cải tiến',

        # Bật/tắt cải tiến
        'use_pruning':            False,  # Optuna MedianPruner
        'use_early_stop':         False,  # Early stopping boosting
        'use_dart':               False,  # DART Dropout booster
        'use_class_weights':      False,  # Cân bằng class mất cân bằng
        'use_focal_loss':         False,  # Focal Loss (tắt tự động nếu bật)
        'use_two_stage':          False,  # Two-Stage Fine-tuning
        'use_weighted_ensemble':  False,  # Weighted Ensemble Prediction
        'use_feat_selection':     False,  # SelectPercentile theo n_samples
        'use_min_rounds':         False,  # Không dừng trước MIN_ROUNDS=40
        'use_scale_lr':           False,  # Scale learning rate theo n_samples
        'use_feat_margin':        False,  # Margin khi chọn Feat vs XGBoost
        'use_ram_gc':             False,  # Giải phóng RAM sau mỗi vòng
        'use_peak_stop':          False,  # Dừng khi chạm ngưỡng peak

        # Tham số
        'n_trials':         100,   # Số trial Optuna
        'n_trials_fine':    0,     # Số trial fine-tune (stage 2)
        'n_boost':          100,   # Số vòng boosting
        'patience':         100,   # Early stop patience (lớn = không dừng)
        'peak_threshold':   1.1,   # Ngưỡng peak (1.1 = không bao giờ đạt)
        'weight_interval':  10,    # Tính weight mỗi N vòng
        'feat_margin':      0.0,   # Margin chọn Feat vs XGBoost
        'dart_rate_drop':   0.1,   # Dropout rate cho DART
        'focal_gamma':      2.0,   # Gamma cho Focal Loss
    },

    # ── M10: Tất cả cải tiến ─────────────────────────────────────────────────
    'm10': {
        'name':        'Full Optimization (M10)',
        'description': 'Tất cả cải tiến — Two-Stage + DART + Class Weights + Weighted Ensemble',

        # Bật/tắt cải tiến
        'use_pruning':            True,   # ✅ Optuna cắt trial kém
        'use_early_stop':         True,   # ✅ Dừng khi không cải thiện
        'use_dart':               True,   # ✅ DART Dropout giảm overfit
        'use_class_weights':      True,   # ✅ Cân bằng class
        'use_focal_loss':         False,  # ❌ Tắt — hại với GOSS
        'use_two_stage':          True,   # ✅ Tìm hyperparams tốt hơn
        'use_weighted_ensemble':  True,   # ✅ Cây tốt đóng góp nhiều hơn
        'use_feat_selection':     True,   # ✅ Giảm chiều features
        'use_min_rounds':         True,   # ✅ Không dừng quá sớm
        'use_scale_lr':           True,   # ✅ LR phù hợp với data size
        'use_feat_margin':        True,   # ✅ Chỉ chọn Feat khi thực sự tốt hơn
        'use_ram_gc':             True,   # ✅ Tiết kiệm RAM
        'use_peak_stop':          True,   # ✅ Dừng khi đạt 99.9%

        # Tham số
        'n_trials':         30,    # Stage 1: khám phá rộng
        'n_trials_fine':    20,    # Stage 2: fine-tune
        'n_boost':          100,   # Số vòng boosting
        'patience':         10,    # Dừng sau 10 lần không cải thiện
        'peak_threshold':   0.999, # Dừng khi đạt 99.9% val_acc
        'weight_interval':  10,    # Tính weight mỗi 10 vòng (cân bằng tốc độ/chính xác)
        'feat_margin':      0.005, # Chỉ chọn Feat nếu tốt hơn XGBoost + 0.5%
        'dart_rate_drop':   0.1,   # Dropout 10% cây
        'focal_gamma':      2.0,   # (không dùng vì focal_loss=False)
    },
}

# ── Hướng dẫn tùy chỉnh ──────────────────────────────────────────────────────
"""
THAM SỐ CÓ THỂ ĐIỀU CHỈNH:

weight_interval:
  1  = Tính weight mỗi vòng → chính xác nhất nhưng CHẬM nhất
  5  = Tính mỗi 5 vòng → cân bằng tốt
  10 = Tính mỗi 10 vòng → mặc định (khuyên dùng)
  20 = Tính mỗi 20 vòng → nhanh nhưng kém chính xác hơn

patience:
  5  = Dừng rất nhanh (dataset đơn giản)
  10 = Mặc định M10
  20 = Chạy lâu hơn, ổn định hơn
  100 = Gần như không dừng sớm

peak_threshold:
  0.990 = Dừng khi đạt 99.0%
  0.999 = Dừng khi đạt 99.9% (mặc định)
  1.1   = Không bao giờ dừng vì peak

feat_margin:
  0.0   = Chọn Feat khi tốt hơn dù chỉ 0.0001
  0.005 = Chỉ chọn Feat khi tốt hơn ít nhất 0.5% (mặc định)
  0.01  = Chỉ chọn Feat khi tốt hơn ít nhất 1%

n_trials + n_trials_fine:
  30 + 20 = Tổng 50 trials, nhanh hơn 100 trials đơn
  50 + 30 = Tổng 80 trials, kết quả tốt hơn
  100 + 0 = Tắt two-stage, dùng 100 trials đơn như baseline
"""


LABEL_TO_NAME = [
    'annealing', 'breast-cancer-wisc-diag', 'breast-cancer-wisc-prog',
    'congressional-voting', 'conn-bench-sonar-mines-rocks',
    'conn-bench-vowel-deterding', 'credit-approval', 'cylinder-bands',
    'dermatology', 'flags', 'heart-cleveland', 'heart-hungarian',
    'heart-va', 'hepatitis', 'horse-colic', 'ionosphere', 'libras',
    'molec-biol-promoter', 'oocytes_merluccius_nucleus_4d',
    'oocytes_merluccius_states_2f', 'oocytes_trisopterus_nucleus_2f',
    'oocytes_trisopterus_states_5b', 'parkinsons', 'planning',
    'primary-tumor', 'spectf', 'statlog-australian-credit',
    'statlog-german-credit', 'statlog-heart', 'statlog-image',
    'statlog-vehicle', 'synthetic-control', 'wine', 'zoo',
    'pittsburg-bridges-REL-L', 'plant-texture',
    'pittsburg-bridges-MATERIAL', 'plant-shape', 'yeast',
    'breast-tissue', 'wine-quality-white', 'glass',
    'pittsburg-bridges-TYPE', 'plant-margin', 'pittsburg-bridges-T-OR-D',
    'bank', 'blood',
]


# ─────────────────────────────────────────────────────────────────────────────
def _build_data_levels(params, n_levels=10):
    """
    Tạo 10 cấp dữ liệu tăng dần.

    Fix lỗi data ít:
    1. Mỗi class cần ít nhất 4 mẫu trong train (để chia val + train)
    2. Tổng train tối thiểu = n_classes * 4
    3. Dùng replace=True khi quá ít mẫu (oversampling)
    """
    dataset_module = importlib.import_module(params['dataset_file'])
    train_x, train_y = dataset_module.get_training_data(
        params['source'], params['class_label'], params['seed']
    )
    test_x,  test_y  = dataset_module.get_testing_data(
        params['source'], params['class_label'], params['seed']
    )
    val_x,   val_y   = dataset_module.get_validation_data(
        params['source'], params['class_label'], params['seed']
    )

    train_x = np.array(train_x)
    train_y = np.array(train_y).ravel()
    test_x  = np.array(test_x)
    val_x   = np.array(val_x)

    total_train = len(train_x)
    n_classes   = len(np.unique(train_y))

    # Mỗi class cần ít nhất 4 mẫu để:
    # - 1 mẫu cho validation
    # - 3 mẫu cho training (XGBoost cần ít nhất vài mẫu/class)
    min_per_class = 4
    min_total     = max(n_classes * min_per_class, 30)

    levels = []
    for i in range(1, n_levels + 1):
        pct      = i / n_levels
        n_sample = int(total_train * pct)

        # Đảm bảo đủ mẫu tối thiểu
        n_sample = max(n_sample, min_total)
        n_sample = min(n_sample, total_train)

        actual_pct = n_sample / total_train * 100

        # Lấy mẫu stratified
        indices = _stratified_sample(train_y, n_sample, min_per_class)

        levels.append({
            'level':        i,
            'pct':          actual_pct,
            'n_sample':     n_sample,
            'total_train':  total_train,
            'n_classes':    n_classes,
            'min_total':    min_total,
            'train_x':      train_x[indices],
            'train_y':      train_y[indices],
            'test_x':       test_x,
            'test_y':       test_y,
            'val_x':        val_x,
            'val_y':        val_y,
        })

        flag = ' ⚡ (tối thiểu)' if n_sample <= min_total else ''
        print(f'  Cấp {i:2d}: {actual_pct:5.1f}% → '
              f'{n_sample}/{total_train} mẫu{flag}')

    return levels


def _stratified_sample(y, n_sample, min_per_class=4):
    """
    Lấy mẫu stratified, đảm bảo mỗi class có ít nhất min_per_class mẫu.
    Dùng replace=True nếu class quá ít mẫu (oversampling).
    """
    classes, counts = np.unique(y, return_counts=True)
    indices = []

    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]

        # Số mẫu muốn lấy cho class này (theo tỷ lệ)
        n_cls = max(min_per_class, int(n_sample * cnt / len(y)))

        if n_cls <= len(cls_idx):
            # Đủ mẫu → lấy không lặp
            chosen = np.random.choice(cls_idx, size=n_cls, replace=False)
        else:
            # Quá ít mẫu → lấy có lặp (oversampling)
            chosen = np.random.choice(cls_idx, size=n_cls, replace=True)

        indices.extend(chosen.tolist())

    # Loại trùng lặp (chỉ loại với các mẫu không oversampled)
    indices = indices[:n_sample]
    return np.array(indices)


# ─────────────────────────────────────────────────────────────────────────────
def _run_one_fold(params, level_data, method_key, method_cfg,
                  output_path, dataset_name):
    """Chạy 1 fold, đo thời gian + bộ nhớ."""
    tracemalloc.start()
    fold_start = time.time()

    try:
        clf = my_featExgboost(
            params=params,
            output_path=output_path,
            data_name=dataset_name,
            level_config=method_cfg,
            override_data={
                'train_x': level_data['train_x'],
                'train_y': level_data['train_y'],
                'test_x':  level_data['test_x'],
                'test_y':  level_data['test_y'],
                'val_x':   level_data['val_x'],
                'val_y':   level_data['val_y'],
            }
        )
        best_cfg = clf.train()
        acc, f1_macro, f1_micro, f1_weight, precision, recall = clf.predict(best_cfg)

        model_choice        = clf.model_choice
        feat_acc_optuna     = clf.feat_acc_optuna
        default_acc_optuna  = clf.default_acc_optuna
        early_stop_at       = clf.early_stop_rounds[-1] \
                              if clf.early_stop_rounds else method_cfg['n_boost']

    except Exception as e:
        import traceback
        print(f'\n      ⚠ Lỗi: {e}')
        traceback.print_exc()
        acc = f1_macro = f1_micro = f1_weight = precision = recall = 0.0
        model_choice       = 'Error'
        feat_acc_optuna    = 0.0
        default_acc_optuna = 0.0
        early_stop_at      = 0

    fold_time = time.time() - fold_start
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'acc':                acc,
        'f1_macro':           f1_macro,
        'f1_micro':           f1_micro,
        'f1_weight':          f1_weight,
        'precision':          precision,
        'recall':             recall,
        'fold_time':          fold_time,
        'mem_peak_mb':        mem_peak / 1024 / 1024,
        'model_choice':       model_choice,
        'feat_acc_optuna':    feat_acc_optuna,
        'default_acc_optuna': default_acc_optuna,
        'early_stop_at':      early_stop_at,
        'method_key':         method_key,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _avg_acc(fold_results):
    valid = [r['acc'] for r in fold_results if r['model_choice'] != 'Error']
    return np.mean(valid) if valid else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(params_file, n_folds=4):
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    params['method'] = 'benchmark'
    dataset_name     = LABEL_TO_NAME[params['class_label']]
    output_path      = expanduservars(params['output_path'] + '//benchmark')
    os.makedirs(output_path, exist_ok=True)

    print(f'\n{"=" * 65}')
    print(f'  Dataset  : {dataset_name}')
    print(f'  Phương án: Baseline | Full Optimization (M10)')
    print(f'  Cấp data : 10 cấp (10% → 100%)')
    print(f'  Số fold  : {n_folds}')
    print(f'{"=" * 65}')

    # Tạo 10 cấp dữ liệu
    print('\n📊 Tạo 10 cấp dữ liệu:')
    params['seed'] = params['seed'] + params['class_label']
    set_seeds(params['seed'])
    data_levels = _build_data_levels(params, n_levels=10)

    all_results = {}

    for level_data in data_levels:
        lvl      = level_data['level']
        pct      = level_data['pct']
        n_sample = level_data['n_sample']
        total    = level_data['total_train']

        print(f'\n{"─" * 65}')
        print(f'  📦 Cấp {lvl}/10 — {pct:.1f}% '
              f'({n_sample}/{total} mẫu | {level_data["n_classes"]} class)')
        print(f'{"─" * 65}')

        all_results[lvl] = {}

        for method_key, method_cfg in METHODS.items():
            print(f'\n  🔧 {method_cfg["name"]}')

            method_start = time.time()
            fold_results = []

            for fold_id in range(n_folds):
                fold_params         = params.copy()
                fold_params['seed'] = fold_id + params['class_label']
                set_seeds(fold_params['seed'])

                print(f'    Fold {fold_id+1}/{n_folds}...', end=' ', flush=True)
                result = _run_one_fold(
                    fold_params, level_data, method_key, method_cfg,
                    output_path, dataset_name
                )
                fold_results.append(result)

                status = '⚠' if result['model_choice'] == 'Error' else '✓'
                print(f'{status} Acc={result["acc"]:.4f}  '
                      f'Model={result["model_choice"]}  '
                      f'Mem={result["mem_peak_mb"]:.1f}MB  '
                      f'Time={result["fold_time"]:.1f}s')

            method_time = time.time() - method_start
            avg_acc     = _avg_acc(fold_results)
            feat_cnt    = sum(1 for r in fold_results
                              if r['model_choice'] == 'Feat-XGBoost')

            print(f'  ✅ {method_cfg["name"]} Cấp {lvl}: '
                  f'Avg Acc={avg_acc:.4f}  '
                  f'Feat={feat_cnt}/{n_folds}  '
                  f'Time={method_time:.1f}s')

            all_results[lvl][method_key] = {
                'method_cfg':   method_cfg,
                'fold_results': fold_results,
                'total_time':   method_time,
                'n_sample':     n_sample,
                'pct':          pct,
            }

    # Xuất Excel + biểu đồ
    print(f'\n{"=" * 65}')
    print('  📝 Xuất kết quả ra Excel + vẽ biểu đồ...')
    excel_path = save_benchmark_to_excel(
        output_path, dataset_name, all_results, data_levels
    )
    print(f'  ✅ Đã xuất: {excel_path}')
    print(f'{"=" * 65}\n')

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
def _find_available_datasets(data_root):
    """Tự động tìm dataset nào đã có đủ 4 file."""
    available = []
    for idx, name in enumerate(LABEL_TO_NAME):
        out_dir  = os.path.join(data_root, name)
        required = [f'{name}.dat', 'label.dat', 'folds.dat', 'validation.dat']
        if all(os.path.exists(os.path.join(out_dir, f)) for f in required):
            available.append((idx, name))
    return available


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PARAMS_FILE = (
        "D:\\XGBoost-by-ensemble-feature-engineering-main\\"
        "XGBoost-by-ensemble-feature-engineering-main\\"
        "featureEng\\src\\config\\params.UCIdata.yml"
    )

    DATA_ROOT = (
        "D:\\XGBoost-by-ensemble-feature-engineering-main\\"
        "XGBoost-by-ensemble-feature-engineering-main\\data"
    )

    # Tìm dataset có sẵn
#    available = _find_available_datasets(DATA_ROOT)
#    print(f'\n📦 Tìm thấy {len(available)} dataset:')
#    for idx, name in available:
#        print(f'   [{idx:2d}] {name}')
#    available = [(41, 'glass'), (39, 'breast-tissue'), (22, 'parkinsons')]
    available = [(1, 'breast-cancer-wisc-diag'), (45, 'bank'), (46, 'blood')]
#    available = [(45, 'bank'), (46, 'blood')]

    # Chạy benchmark từng dataset
    for idx, name in available:
        print(f'\n{"#" * 65}')
        print(f'  🚀 [{idx}] {name}')
        print(f'{"#" * 65}')

        tmp_file = None
        try:
            with open(PARAMS_FILE, 'r') as f:
                params = yaml.safe_load(f)
            params['class_label'] = idx

            tmp_file = PARAMS_FILE.replace('.yml', '_tmp.yml')
            with open(tmp_file, 'w') as f:
                yaml.dump(params, f)

            run_benchmark(tmp_file, n_folds=4)

        except Exception as e:
            import traceback
            print(f'  ⚠ Lỗi [{idx}] {name}: {e}')
            traceback.print_exc()
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    print(f'\n🎉 Hoàn tất {len(available)} dataset!')