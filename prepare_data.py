import os
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split

def create_uci_format_dataset():
    # 1. Tải dữ liệu wine (Vị trí 'wine' trong label_to_name của tác giả)
    data = load_wine()
    X = data.data
    y = data.target

    # 2. Tạo thư mục chứa data
    dataset_name = 'wine'
    out_dir = f'./data/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)

    # 3. Lưu data và label ra file .dat
    np.savetxt(f'{out_dir}/{dataset_name}.dat', X, delimiter=',')
    np.savetxt(f'{out_dir}/label.dat', y, delimiter=',')

    # 4. Tạo ma trận folds và validation (Tác giả code hardcode chia 4 folds)
    n_samples = X.shape[0]
    n_folds = 4
    folds_index = np.zeros((n_samples, n_folds))
    validation = np.zeros((n_samples, n_folds))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for cv_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        # Đánh dấu tập test (bằng 1)
        folds_index[test_idx, cv_idx] = 1

        # Chia phần còn lại thành train và validation (ví dụ lấy 10% làm val)
        y_train_val = y[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.1, random_state=42, stratify=y_train_val
        )

        # Đánh dấu tập validation (bằng 1)
        validation[val_idx, cv_idx] = 1

    # Lưu 2 file ma trận cấu trúc
    np.savetxt(f'{out_dir}/folds.dat', folds_index, delimiter=',', fmt='%d')
    np.savetxt(f'{out_dir}/validation.dat', validation, delimiter=',', fmt='%d')

    print(f"Đã tạo thành công cấu trúc dữ liệu '{dataset_name}' tại {out_dir}")

if __name__ == '__main__':
    create_uci_format_dataset()