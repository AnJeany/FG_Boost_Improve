import logging
from typing import Union
import os
import shutil
import random
import importlib
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from torch import nn
import torch

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

LOGGER = logging.getLogger(__name__)
ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]

# ─── Màu sắc ────────────────────────────────────────────────────────────────
CLR_HEADER      = '2F5496'
CLR_TITLE       = '1F3864'
CLR_AVG         = 'D9E1F2'
CLR_TIME        = 'E2EFDA'
CLR_COMPARE_HDR = '375623'
CLR_COMPARE_ROW = 'EBF1DE'
CLR_BEST        = 'FFFF00'
CLR_FEAT        = 'C6EFCE'   # xanh lá — Feat-XGBoost
CLR_DEFAULT     = 'FFEB9C'   # vàng nhạt — XGBoost gốc
CLR_ERROR       = 'FFC7CE'   # đỏ nhạt — Error
CLR_RUN_HEADER  = '4472C4'   # xanh — header lần chạy


def _make_styles():
    thin   = Side(style='thin')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal='center', vertical='center')
    white_bold = Font(name='Arial', bold=True, color='FFFFFF', size=11)
    return border, center, white_bold


def _auto_width(ws):
    for col_cells in ws.columns:
        max_len    = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            try:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            except:
                pass
        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)


def _reorder_sheets(wb):
    priority_last = ['Comparison', 'Timing Summary']
    ordered = [s for s in wb.sheetnames if s not in priority_last] + \
              [s for s in priority_last if s in wb.sheetnames]
    for i, name in enumerate(ordered):
        wb.move_sheet(name, offset=i - wb.sheetnames.index(name))


def _model_choice_color(choice):
    if choice == 'Feat-XGBoost':
        return CLR_FEAT
    elif choice == 'XGBoost':
        return CLR_DEFAULT
    return CLR_ERROR


def _get_run_number(wb, dataset_name):
    """Đếm số sheet có tên bắt đầu bằng dataset_name để tính số lần chạy."""
    count = sum(1 for s in wb.sheetnames if s.startswith(dataset_name[:20]))
    return count + 1


# ─────────────────────────────────────────────────────────────────────────────
def save_results_to_excel(output_path, dataset_name, fold_results, timing_info):
    """
    Lưu vào results_all.xlsx:
      - Mỗi lần chạy tạo 1 sheet mới: wine_Run1_09-30, wine_Run2_10-15, ...
      - Sheet Comparison: so sánh tất cả dataset (lần chạy mới nhất)
      - Sheet Timing Summary: lịch sử thời gian tất cả lần chạy
    """
    excel_path = os.path.join(output_path, 'results_all.xlsx')
    border, center, white_bold = _make_styles()

    wb = load_workbook(excel_path) if os.path.exists(excel_path) else Workbook()
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])

    n_folds    = len(fold_results)
    timestamp  = datetime.now().strftime('%m-%d_%H-%M')
    run_number = _get_run_number(wb, dataset_name[:20])

    # Tên sheet: wine_Run1_04-06_09-30 (giới hạn 31 ký tự)
    raw_sheet_name = f'{dataset_name}_Run{run_number}_{timestamp}'
    sheet_name     = raw_sheet_name[:31]

    # ── 1. Sheet mới cho lần chạy này ───────────────────────────────────────
    ws = wb.create_sheet(sheet_name)

    # Tiêu đề chính
    ws.merge_cells('A1:J1')
    ws['A1'].value     = (f'Dataset: {dataset_name}  |  '
                          f'Lần chạy #{run_number}  |  '
                          f'{datetime.now().strftime("%Y-%m-%d %H:%M")}')
    ws['A1'].font      = Font(name='Arial', bold=True, size=13, color='FFFFFF')
    ws['A1'].fill      = PatternFill('solid', start_color=CLR_TITLE)
    ws['A1'].alignment = center
    ws.row_dimensions[1].height = 28

    # Chú thích màu
    ws.merge_cells('A2:J2')
    ws['A2'].value     = ('🟢 Xanh = Feat-XGBoost được chọn    '
                          '🟡 Vàng = XGBoost gốc được chọn    '
                          '🔴 Đỏ = Lỗi')
    ws['A2'].font      = Font(name='Arial', italic=True, size=10)
    ws['A2'].alignment = center
    ws.row_dimensions[2].height = 18

    # Header cột
    headers = [
        'Fold',
        'Model được chọn',
        'Val Acc (Feat)',
        'Val Acc (XGB)',
        'Test Accuracy',
        'F1 Macro',
        'F1 Micro',
        'F1 Weight',
        'Precision',
        'Recall',
        'Thời gian (s)',
    ]
    for col, h in enumerate(headers, 1):
        cell           = ws.cell(row=3, column=col, value=h)
        cell.font      = white_bold
        cell.fill      = PatternFill('solid', start_color=CLR_RUN_HEADER)
        cell.alignment = center
        cell.border    = border
    ws.row_dimensions[3].height = 20

    # Dữ liệu từng fold
    for i, res in enumerate(fold_results):
        row          = i + 4
        choice       = res.get('model_choice', 'N/A')
        row_color    = _model_choice_color(choice)
        fold_time    = res.get('fold_time', 0)
        feat_acc_opt = res.get('feat_acc_optuna', 0)
        def_acc_opt  = res.get('default_acc_optuna', 0)

        values = [
            f'Fold {i+1}',
            choice,
            feat_acc_opt,
            def_acc_opt,
            res['acc'],
            res['f1_macro'],
            res['f1_micro'],
            res['f1_weight'],
            res['precision'],
            res['recall'],
            round(fold_time, 2),
        ]
        for col, val in enumerate(values, 1):
            cell           = ws.cell(row=row, column=col, value=val)
            cell.fill      = PatternFill('solid', start_color=row_color)
            cell.alignment = center
            cell.border    = border
            if col > 2:
                cell.number_format = '0.0000' if col < 11 else '0.00'

    # Hàng Average
    avg_row = n_folds + 4
    ws.cell(row=avg_row, column=1, value='Average').font      = Font(bold=True)
    ws.cell(row=avg_row, column=1).alignment = center
    ws.cell(row=avg_row, column=1).border    = border
    ws.cell(row=avg_row, column=2, value='-').alignment = center
    ws.cell(row=avg_row, column=2).border = border

    for col in range(3, 12):
        col_letter         = get_column_letter(col)
        cell               = ws.cell(row=avg_row, column=col)
        cell.value         = f'=AVERAGE({col_letter}4:{col_letter}{avg_row-1})'
        cell.fill          = PatternFill('solid', start_color=CLR_AVG)
        cell.alignment     = center
        cell.border        = border
        cell.font          = Font(bold=True)
        cell.number_format = '0.0000' if col < 11 else '0.00'

    # Đếm số fold chọn Feat vs XGBoost
    feat_count    = sum(1 for r in fold_results if r.get('model_choice') == 'Feat-XGBoost')
    default_count = sum(1 for r in fold_results if r.get('model_choice') == 'XGBoost')
    summary_row   = avg_row + 2

    ws.merge_cells(f'A{summary_row}:J{summary_row}')
    ws[f'A{summary_row}'].value = (
        f'📊 Tổng kết lựa chọn model: '
        f'Feat-XGBoost={feat_count}/{n_folds} fold  |  '
        f'XGBoost gốc={default_count}/{n_folds} fold  |  '
        f'⏱ Tổng thời gian: {timing_info.get("total_time", 0):.1f}s'
    )
    ws[f'A{summary_row}'].font      = Font(name='Arial', bold=True, size=11)
    ws[f'A{summary_row}'].alignment = center
    ws[f'A{summary_row}'].fill      = PatternFill('solid', start_color=CLR_AVG)

    _auto_width(ws)

    # ── 2. Sheet Comparison (cập nhật lần chạy mới nhất của mỗi dataset) ────
    if 'Comparison' not in wb.sheetnames:
        ws_cmp = wb.create_sheet('Comparison')
        ws_cmp.merge_cells('A1:I1')
        ws_cmp['A1'].value     = 'So sánh kết quả giữa các Dataset (lần chạy mới nhất)'
        ws_cmp['A1'].font      = Font(name='Arial', bold=True, size=13, color='FFFFFF')
        ws_cmp['A1'].fill      = PatternFill('solid', start_color=CLR_COMPARE_HDR)
        ws_cmp['A1'].alignment = center
        ws_cmp.row_dimensions[1].height = 28

        cmp_headers = ['Dataset', 'Lần chạy', 'Avg Accuracy', 'Avg F1 Macro',
                       'Avg F1 Micro', 'Avg F1 Weight', 'Avg Precision',
                       'Avg Recall', 'Tổng thời gian (s)']
        for col, h in enumerate(cmp_headers, 1):
            cell           = ws_cmp.cell(row=2, column=col, value=h)
            cell.font      = white_bold
            cell.fill      = PatternFill('solid', start_color=CLR_COMPARE_HDR)
            cell.alignment = center
            cell.border    = border
    else:
        ws_cmp = wb['Comparison']

    # Tìm hàng dataset hoặc thêm mới
    write_row = None
    for row in ws_cmp.iter_rows(min_row=3):
        if row[0].value == dataset_name:
            write_row = row[0].row
            break
    if write_row is None:
        write_row = ws_cmp.max_row + 1
        if write_row < 3:
            write_row = 3

    # Dùng formula tham chiếu sang sheet lần chạy mới nhất
    row_data = [
        dataset_name,
        f'Run #{run_number}',
    ] + [
        f"='{sheet_name}'!{get_column_letter(c)}{avg_row}"
        for c in range(5, 12)  # cột Test Accuracy → Recall (cột 5-11)
    ]
    for col, val in enumerate(row_data, 1):
        cell               = ws_cmp.cell(row=write_row, column=col, value=val)
        cell.fill          = PatternFill('solid', start_color=CLR_COMPARE_ROW)
        cell.alignment     = center
        cell.border        = border
        if col > 2:
            cell.number_format = '0.0000' if col < 9 else '0.00'

    _auto_width(ws_cmp)

    # ── 3. Sheet Timing Summary (lịch sử tất cả lần chạy) ───────────────────
    if 'Timing Summary' not in wb.sheetnames:
        ws_time = wb.create_sheet('Timing Summary')
        ws_time.merge_cells('A1:F1')
        ws_time['A1'].value     = 'Lịch sử thời gian chạy'
        ws_time['A1'].font      = Font(name='Arial', bold=True, size=13, color='FFFFFF')
        ws_time['A1'].fill      = PatternFill('solid', start_color=CLR_TITLE)
        ws_time['A1'].alignment = center
        ws_time.row_dimensions[1].height = 28

        time_headers = ['Dataset', 'Lần chạy', 'Tổng thời gian (s)',
                        'TB mỗi fold (s)', 'Số fold', 'Thời điểm chạy']
        for col, h in enumerate(time_headers, 1):
            cell           = ws_time.cell(row=2, column=col, value=h)
            cell.font      = white_bold
            cell.fill      = PatternFill('solid', start_color=CLR_HEADER)
            cell.alignment = center
            cell.border    = border
    else:
        ws_time = wb['Timing Summary']

    # Luôn thêm hàng mới (không ghi đè — giữ lịch sử)
    time_write_row = ws_time.max_row + 1
    if time_write_row < 3:
        time_write_row = 3

    total_time = timing_info.get('total_time', 0)
    avg_time   = total_time / n_folds if n_folds > 0 else 0

    for col, val in enumerate([
        dataset_name,
        f'Run #{run_number}',
        round(total_time, 2),
        round(avg_time, 2),
        n_folds,
        datetime.now().strftime('%Y-%m-%d %H:%M'),
    ], 1):
        cell           = ws_time.cell(row=time_write_row, column=col, value=val)
        cell.fill      = PatternFill('solid', start_color=CLR_TIME)
        cell.alignment = center
        cell.border    = border

    _auto_width(ws_time)

    # Sắp xếp sheet
    _reorder_sheets(wb)

    wb.save(excel_path)
    print(f'\n✅ Đã lưu vào: {excel_path}  (sheet: {sheet_name})')
    return excel_path


# ─────────────────────────────────────────────────────────────────────────────
def _build_datasets(params):
    dataset_file   = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_x, train_y = dataset_module.get_training_data(params['source'], params["class_label"], params["seed"])
    test_x, test_y   = dataset_module.get_testing_data(params['source'], params["class_label"], params["seed"])
    space            = dataset_module.get_space()
    return train_x, train_y, test_x, test_y, space


def _build_datasets_UCI(params):
    dataset_file   = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_x, train_y = dataset_module.get_training_data(params['source'], params["class_label"], params["seed"])
    test_x, test_y   = dataset_module.get_testing_data(params['source'], params["class_label"], params["seed"])
    val_x, val_y     = dataset_module.get_validation_data(params['source'], params["class_label"], params["seed"])
    space            = dataset_module.get_space()
    return train_x, train_y, test_x, test_y, val_x, val_y, space


def kfold_split(train_x, train_y, fold, params):
    kf = StratifiedKFold(n_splits=params["folds"], shuffle=True, random_state=params["seed"]) \
         if params["stratified"] else \
         KFold(n_splits=params["folds"], shuffle=True, random_state=params["seed"])
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        if i == fold:
            return train_x[train_index], train_y[train_index], train_x[test_index], train_y[test_index]


class WithStateDict(nn.Module):
    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str, params_file: str) -> None:
    shutil.copy(params_file, path)
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_results_cc(output_path, params, f1_micro, f1_macro, f1_weight, acc, precision, recall):
    for name, val in [('micro', f1_micro), ('macro', f1_macro), ('f1weight', f1_weight),
                      ('acc', acc), ('precision', precision), ('recall', recall)]:
        with open(f"{output_path}/results_{name}.txt", "a") as f:
            f.write(str(val) + '\n')


def save_results(output_path, params, f1_micro, f1_macro, f1_weight, acc):
    f1_micro  /= params["folds"]
    f1_macro  /= params["folds"]
    f1_weight /= params["folds"]
    acc       /= params["folds"]
    for name, val in [('micro', f1_micro), ('macro', f1_macro),
                      ('f1weight', f1_weight), ('acc', acc)]:
        with open(f"{output_path}/results_{name}.txt", "a") as f:
            f.write(str(val) + ' ')


def save_results_acc(output_path, params, acc, cv):
    acc /= params["folds"]
    with open(output_path + "/results_acc-" + str(cv) + ".txt", "a") as f:
        f.write(str(acc) + '\n')