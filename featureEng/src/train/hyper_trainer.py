import os
import time
import joblib
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from .utils import save_results_cc, save_results_to_excel, expanduservars, _build_datasets_UCI
from ..fEngeering import transformData
from src.train import tuneEngineeringXGBv31, tuneEngAssistXGBAv51
import pandas as pd

from pathlib import Path

label_to_name = ['annealing',
 'breast-cancer-wisc-diag',
 'breast-cancer-wisc-prog',
 'congressional-voting',
 'conn-bench-sonar-mines-rocks',
 'conn-bench-vowel-deterding',
 'credit-approval',
 'cylinder-bands',
 'dermatology',
 'flags',
 'heart-cleveland',
 'heart-hungarian',
 'heart-va',
 'hepatitis',
 'horse-colic',
 'ionosphere',
 'libras',
 'molec-biol-promoter',
 'oocytes_merluccius_nucleus_4d',
 'oocytes_merluccius_states_2f',
 'oocytes_trisopterus_nucleus_2f',
 'oocytes_trisopterus_states_5b',
 'parkinsons',
 'planning',
 'primary-tumor',
 'spectf',
 'statlog-australian-credit',
 'statlog-german-credit',
 'statlog-heart',
 'statlog-image',
 'statlog-vehicle',
 'synthetic-control',
 'wine',
 'zoo',
 'pittsburg-bridges-REL-L',
 'plant-texture',
 'pittsburg-bridges-MATERIAL',
 'plant-shape',
 'yeast',
 'breast-tissue',
 'wine-quality-white',
 'glass',
 'pittsburg-bridges-TYPE',
 'plant-margin',
 'pittsburg-bridges-T-OR-D',
 'bank',
 'blood',
]


def get_param(args=None):
    return {
        'subsample':         args['subsample'],
        'colsample_bylevel': args['colsample_bylevel'],
        'colsample_bynode':  args['colsample_bynode'],
        'colsample_bytree':  args['colsample_bytree'],
        'eta':               args['eta'],
        'gamma':             args['gamma'],
        'max_depth':         args['max_depth'],
        'n_estimators':      args['round'],
    }


def run_feat_UCI(params: dict, params_file):
    output_path = expanduservars(params['output_path'] + '//' + params['method'])
    os.makedirs(output_path, exist_ok=True)

    for gassid in [params['class_label']]:
        params['class_label'] = gassid
        dataset_name = label_to_name[gassid]

        print(f'\n{"=" * 60}')
        print(f'Dataset: {dataset_name}  (index={gassid})')
        print(f'{"=" * 60}')

        hyp_acc       = []
        hyp_f1_micro  = []
        hyp_f1_macro  = []
        hyp_f1_weight = []
        hyp_precision = []
        hyp_recall    = []
        fold_results  = []
        fold_times    = []

        total_start = time.time()

        for seedid in range(4):
            params['seed'] = seedid
            fold_start = time.time()
            print(f'\n  ▶ Fold {seedid + 1}/4 — seed={seedid}')

            classifier = tuneEngAssistXGBAv51.my_featExgboost(
                params=params,
                output_path=output_path,
                data_name=dataset_name
            )

            try:
                best_config = classifier.train()
                acc, f1_macro, f1_micro, f1_weight, precision, recall = classifier.predict(best_config)
                model_choice       = classifier.model_choice
                feat_acc_optuna    = classifier.feat_acc_optuna
                default_acc_optuna = classifier.default_acc_optuna
            except Exception as e:
                print(f'  ⚠ Lỗi fold {seedid + 1}: {e}')
                acc = f1_macro = f1_micro = f1_weight = precision = recall = 0
                model_choice       = 'Error'
                feat_acc_optuna    = 0.0
                default_acc_optuna = 0.0

            fold_time = time.time() - fold_start
            fold_times.append(fold_time)

            print(f'  ✔ Fold {seedid + 1} — Acc={acc:.4f}  '
                  f'F1_macro={f1_macro:.4f}  '
                  f'Model={model_choice}  '
                  f'Thời gian={fold_time:.1f}s')

            hyp_acc.append(acc)
            hyp_f1_micro.append(f1_micro)
            hyp_f1_macro.append(f1_macro)
            hyp_f1_weight.append(f1_weight)
            hyp_precision.append(precision)
            hyp_recall.append(recall)

            fold_results.append({
                'acc':                acc,
                'f1_macro':           f1_macro,
                'f1_micro':           f1_micro,
                'f1_weight':          f1_weight,
                'precision':          precision,
                'recall':             recall,
                'fold_time':          fold_time,
                'model_choice':       model_choice,
                'feat_acc_optuna':    feat_acc_optuna,
                'default_acc_optuna': default_acc_optuna,
            })

        total_time = time.time() - total_start

        print(f'\n  📊 Kết quả trung bình ({dataset_name}):')
        print(f'     Accuracy  : {np.mean(hyp_acc):.4f} ± {np.std(hyp_acc):.4f}')
        print(f'     F1 Macro  : {np.mean(hyp_f1_macro):.4f}')
        print(f'     F1 Weight : {np.mean(hyp_f1_weight):.4f}')
        print(f'     ⏱ Tổng   : {total_time:.1f}s  (TB/fold: {total_time/4:.1f}s)')

        save_results_cc(output_path, params,
                        hyp_f1_micro, hyp_f1_macro, hyp_f1_weight,
                        hyp_acc, hyp_precision, hyp_recall)

        timing_info = {'fold_times': fold_times, 'total_time': total_time}
        save_results_to_excel(output_path, dataset_name, fold_results, timing_info)


def run_feat_save(params: dict, params_file):
    output_path = expanduservars('Featdata' + '//')
    os.makedirs(output_path, exist_ok=True)

    for gassid in range(len(label_to_name)):
        dataset_name = label_to_name[gassid]

        # ✅ CHECK FILE .dat
        data_files = list(Path("data").rglob(f"{dataset_name}.dat"))
        if len(data_files) == 0:
            print(f"⏭ Skip {dataset_name} (không có .dat)")
            continue

        params.update({'class_label': gassid})
        print('this is ', dataset_name, ' -start')

        for seedid in range(4):
            params['seed'] = seedid
            train_x, train_y, test_x, test_y, val_x, val_y, space = _build_datasets_UCI(params)
            trs = transformData(train_x, train_y.ravel(), val_x, test_x)

            try:
                x_new, x_test_new, x_val_new = trs._transform_autofeat()
            except:
                x_new, x_test_new, x_val_new = (
                    pd.DataFrame(train_x), pd.DataFrame(test_x), pd.DataFrame(val_x)
                )
            save_data(os.path.join(output_path, label_to_name[gassid]),
                      (x_new, x_test_new, x_val_new), splits=('auto' + str(seedid)))

            for method, transform in [
                ('hpca',     trs._transform_hpca),
                ('randP',    trs._transform_randomProject),
                ('minmax',   trs._transform_minmax),
                ('robuster', trs._transform_robuster),
            ]:
                x_new, x_test_new, x_val_new = transform()
                save_data(os.path.join(output_path, label_to_name[gassid]),
                          (x_new, x_test_new, x_val_new), splits=(method + str(seedid)))


def save_data(path, data, splits='1'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(data, os.path.join(path, splits + '_data.pkl'))


def load_data(path, splits='1'):
    return joblib.load(os.path.join(path, splits + '_data.pkl'))


def get_score(test_y, test_y_pred):
    f1_micro  = f1_score(test_y, test_y_pred, average="micro")
    f1_macro  = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc       = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred, average="macro", zero_division=0)
    recall    = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precision, recall