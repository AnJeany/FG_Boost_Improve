"""
download_datasets.py
Tự động tải và chuẩn bị tất cả 45 dataset từ UCI ML Repository.

Cách dùng:
    pip install ucimlrepo
    python featureEng/download_datasets.py

Kết quả: tạo thư mục data/<tên_dataset>/ với các file:
    - <tên>.dat
    - label.dat
    - folds.dat
    - validation.dat
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── Mapping 45 dataset: tên trong code → UCI ID ────────────────────────────
# UCI ID lấy từ https://archive.ics.uci.edu
DATASET_MAP = {
    # index: (tên_folder, uci_id, tên_hiển_thị)
    0:  ('annealing',                    'annealing',              53),
    1:  ('breast-cancer-wisc-diag',      'breast-cancer-wisc-diag', 17),
    2:  ('breast-cancer-wisc-prog',      'breast-cancer-wisc-prog', 16),
    3:  ('congressional-voting',         'congressional-voting',    105),
    4:  ('conn-bench-sonar-mines-rocks', 'conn-bench-sonar-mines-rocks', 151),
    5:  ('conn-bench-vowel-deterding',   'conn-bench-vowel-deterding', 152),
    6:  ('credit-approval',              'credit-approval',         27),
    7:  ('cylinder-bands',               'cylinder-bands',          32),
    8:  ('dermatology',                  'dermatology',             33),
    9:  ('flags',                        'flags',                   40),
    10: ('heart-cleveland',              'heart-cleveland',         45),
    11: ('heart-hungarian',              'heart-hungarian',         48),
    12: ('heart-va',                     'heart-va',                200),
    13: ('hepatitis',                    'hepatitis',               46),
    14: ('horse-colic',                  'horse-colic',             47),
    15: ('ionosphere',                   'ionosphere',              52),
    16: ('libras',                       'libras',                  144),
    17: ('molec-biol-promoter',          'molec-biol-promoter',    67),
    18: ('oocytes_merluccius_nucleus_4d','oocytes_merluccius_nucleus_4d', None),
    19: ('oocytes_merluccius_states_2f', 'oocytes_merluccius_states_2f',  None),
    20: ('oocytes_trisopterus_nucleus_2f','oocytes_trisopterus_nucleus_2f', None),
    21: ('oocytes_trisopterus_states_5b','oocytes_trisopterus_states_5b',  None),
    22: ('parkinsons',                   'parkinsons',              174),
    23: ('planning',                     'planning',                230),
    24: ('primary-tumor',                'primary-tumor',           83),
    25: ('spectf',                       'spectf',                  96),
    26: ('statlog-australian-credit',    'statlog-australian-credit', 143),
    27: ('statlog-german-credit',        'statlog-german-credit',   144),
    28: ('statlog-heart',                'statlog-heart',           145),
    29: ('statlog-image',                'statlog-image',           147),
    30: ('statlog-vehicle',              'statlog-vehicle',         149),
    31: ('synthetic-control',            'synthetic-control',       97),
    32: ('wine',                         'wine',                    109),
    33: ('zoo',                          'zoo',                     111),
    34: ('pittsburg-bridges-REL-L',      'pittsburg-bridges-REL-L', None),
    35: ('plant-texture',                'plant-texture',           None),
    36: ('pittsburg-bridges-MATERIAL',   'pittsburg-bridges-MATERIAL', None),
    37: ('plant-shape',                  'plant-shape',             None),
    38: ('yeast',                        'yeast',                   110),
    39: ('breast-tissue',                'breast-tissue',           192),
    40: ('wine-quality-white',           'wine-quality-white',      186),
    41: ('glass',                        'glass',                   42),
    42: ('pittsburg-bridges-TYPE',       'pittsburg-bridges-TYPE',  None),
    43: ('plant-margin',                 'plant-margin',            None),
    44: ('pittsburg-bridges-T-OR-D',     'pittsburg-bridges-T-OR-D', None),
}

# UCI IDs chính xác (dùng ucimlrepo)
UCI_IDS = {
    'annealing':                    53,
    'breast-cancer-wisc-diag':      17,
    'breast-cancer-wisc-prog':      16,
    'congressional-voting':         105,
    'conn-bench-sonar-mines-rocks': 151,
    'conn-bench-vowel-deterding':   152,
    'credit-approval':              27,
    'cylinder-bands':               32,
    'dermatology':                  33,
    'flags':                        40,
    'heart-cleveland':              45,
    'heart-hungarian':              48,
    'heart-va':                     200,
    'hepatitis':                    46,
    'horse-colic':                  47,
    'ionosphere':                   52,
    'libras':                       144,
    'molec-biol-promoter':          67,
    'parkinsons':                   174,
    'planning':                     230,
    'primary-tumor':                83,
    'spectf':                       96,
    'statlog-australian-credit':    143,
    'statlog-german-credit':        144,
    'statlog-heart':                145,
    'statlog-image':                147,
    'statlog-vehicle':              149,
    'synthetic-control':            97,
    'wine':                         109,
    'zoo':                          111,
    'yeast':                        110,
    'breast-tissue':                192,
    'wine-quality-white':           186,
    'glass':                        42,
}


def _prepare_and_save(name, X, y, output_dir, n_folds=4, seed=42):
    """
    Chuẩn bị và lưu dataset theo đúng định dạng project:
    - <name>.dat    : features
    - label.dat     : nhãn
    - folds.dat     : ma trận chia fold (test)
    - validation.dat: ma trận validation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Encode labels về số nguyên 0, 1, 2, ...
    le = LabelEncoder()
    y  = le.fit_transform(y.ravel())
    X  = np.array(X, dtype=float)

    # Xử lý NaN
    col_means = np.nanmean(X, axis=0)
    nan_mask  = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    n_samples = X.shape[0]

    # Lưu features và labels
    np.savetxt(os.path.join(output_dir, f'{name}.dat'), X, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'label.dat'), y, delimiter=',', fmt='%d')

    # Tạo folds.dat và validation.dat
    folds_index = np.zeros((n_samples, n_folds), dtype=int)
    val_index   = np.zeros((n_samples, n_folds), dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for cv_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        folds_index[test_idx, cv_idx] = 1

        y_tv = y[train_val_idx]
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.1,
                random_state=seed, stratify=y_tv
            )
        except ValueError:
            # Nếu class quá ít, không stratify
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.1, random_state=seed
            )
        val_index[val_idx, cv_idx] = 1

    np.savetxt(os.path.join(output_dir, 'folds.dat'),
               folds_index, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(output_dir, 'validation.dat'),
               val_index, delimiter=',', fmt='%d')

    print(f'    ✅ {name}: {n_samples} mẫu, {X.shape[1]} features, '
          f'{len(np.unique(y))} classes')
    return True


def download_with_ucimlrepo(name, uci_id, output_dir):
    """Tải dataset bằng thư viện ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=uci_id)
        X = dataset.data.features
        y = dataset.data.targets
        return _prepare_and_save(name, X.values, y.values, output_dir)
    except Exception as e:
        print(f'    ⚠ ucimlrepo thất bại: {e}')
        return False


def download_with_sklearn(name, output_dir):
    """Tải một số dataset phổ biến bằng sklearn."""
    try:
        from sklearn import datasets
        loaders = {
            'wine':       datasets.load_wine,
            'ionosphere': None,
            'glass':      None,
        }
        if name == 'wine':
            data = datasets.load_wine()
            return _prepare_and_save(name, data.data, data.target, output_dir)
    except Exception as e:
        print(f'    ⚠ sklearn thất bại: {e}')
    return False


def download_all(data_root, indices=None):
    """
    Tải tất cả dataset (hoặc subset theo indices).

    Parameters
    ----------
    data_root : đường dẫn thư mục data gốc
    indices   : list các index muốn tải, None = tải tất cả
    """
    print(f'\n{"=" * 60}')
    print(f'  Tải 45 dataset UCI')
    print(f'  Lưu vào: {data_root}')
    print(f'{"=" * 60}')

    success = []
    failed  = []
    skipped = []

    targets = indices if indices else list(DATASET_MAP.keys())

    for idx in targets:
        info = DATASET_MAP[idx]
        name     = info[0]
        uci_id   = info[2]
        out_dir  = os.path.join(data_root, name)

        print(f'\n[{idx:2d}] {name}')

        # Kiểm tra đã có chưa
        required = [f'{name}.dat', 'label.dat', 'folds.dat', 'validation.dat']
        if all(os.path.exists(os.path.join(out_dir, f)) for f in required):
            print(f'    ⏭ Đã có sẵn, bỏ qua')
            skipped.append(name)
            continue

        # Thử tải bằng ucimlrepo
        if uci_id and download_with_ucimlrepo(name, uci_id, out_dir):
            success.append(name)
            continue

        # Thử bằng sklearn cho một số dataset
        if download_with_sklearn(name, out_dir):
            success.append(name)
            continue

        # Không tải được
        print(f'    ❌ Không tải được tự động — xem hướng dẫn thủ công bên dưới')
        failed.append((idx, name))

    # Tổng kết
    print(f'\n{"=" * 60}')
    print(f'  ✅ Thành công : {len(success)} dataset')
    print(f'  ⏭ Đã có sẵn : {len(skipped)} dataset')
    print(f'  ❌ Thất bại  : {len(failed)} dataset')

    if failed:
        print(f'\n  📋 Các dataset cần tải thủ công:')
        print(f'  {"─" * 50}')
        manual_list = {
            'oocytes_merluccius_nucleus_4d':  'https://archive.ics.uci.edu/dataset/180',
            'oocytes_merluccius_states_2f':   'https://archive.ics.uci.edu/dataset/181',
            'oocytes_trisopterus_nucleus_2f': 'https://archive.ics.uci.edu/dataset/182',
            'oocytes_trisopterus_states_5b':  'https://archive.ics.uci.edu/dataset/183',
            'pittsburg-bridges-REL-L':        'https://archive.ics.uci.edu/dataset/18',
            'pittsburg-bridges-MATERIAL':     'https://archive.ics.uci.edu/dataset/18',
            'pittsburg-bridges-TYPE':         'https://archive.ics.uci.edu/dataset/18',
            'pittsburg-bridges-T-OR-D':       'https://archive.ics.uci.edu/dataset/18',
            'plant-texture':                  'https://archive.ics.uci.edu/dataset/241',
            'plant-shape':                    'https://archive.ics.uci.edu/dataset/241',
            'plant-margin':                   'https://archive.ics.uci.edu/dataset/241',
        }
        for idx, name in failed:
            url = manual_list.get(name, 'https://archive.ics.uci.edu')
            print(f'  [{idx:2d}] {name:<40} → {url}')

    print(f'{"=" * 60}\n')
    return success, failed, skipped


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_ROOT = (
        "D:\\XGBoost-by-ensemble-feature-engineering-main\\"
        "XGBoost-by-ensemble-feature-engineering-main\\data"
    )

    # Cài ucimlrepo nếu chưa có
    try:
        import ucimlrepo
    except ImportError:
        print('Đang cài ucimlrepo...')
        os.system(f'{sys.executable} -m pip install ucimlrepo')
        import ucimlrepo

    # Tải tất cả dataset
    # Hoặc chỉ tải một số: indices=[32, 33, 41] (wine, zoo, glass)
    download_all(DATA_ROOT, indices=None)