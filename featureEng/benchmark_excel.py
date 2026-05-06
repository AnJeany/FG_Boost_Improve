"""
benchmark_excel.py
Thêm cột RAM (Avg/Peak) vào Overview và Charts để so sánh.
"""
import os
import numpy as np
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.chart import LineChart, Reference

C_TITLE  = '1F3864'
C_BASE   = '7F7F7F'
C_M10    = '375623'
C_BASE_L = 'EDEDED'
C_M10_L  = 'E2EFDA'
C_HDR    = '2F5496'
C_AVG    = 'D9E1F2'
C_FEAT   = 'C6EFCE'
C_XGB    = 'FFEB9C'
C_ERROR  = 'FFC7CE'
C_WHITE  = 'FFFFFF'
C_GREY   = 'F2F2F2'
C_WIN    = 'C6EFCE'
C_LOSE   = 'FFC7CE'
C_TIE    = 'FFEB9C'
C_RAM    = 'FCE4D6'   # cam nhạt — RAM

METHOD_NAMES = {
    'baseline': 'Baseline',
    'm10':      'Full Optimization (M10)',
}


def _b():
    s = Side(style='thin')
    return Border(left=s, right=s, top=s, bottom=s)

def _c():
    return Alignment(horizontal='center', vertical='center', wrap_text=True)

def _wb(size=10):
    return Font(name='Arial', bold=True, color=C_WHITE, size=size)

def _fill(color):
    return PatternFill('solid', start_color=color)

def _title_row(ws, text, n_cols, color, row=1):
    ws.merge_cells(f'A{row}:{get_column_letter(n_cols)}{row}')
    c = ws[f'A{row}']
    c.value, c.font, c.fill, c.alignment = (
        text,
        Font(name='Arial', bold=True, size=13, color=C_WHITE),
        _fill(color), _c()
    )
    ws.row_dimensions[row].height = 28

def _auto_w(ws, mn=10, mx=30):
    for col in ws.columns:
        cl  = get_column_letter(col[0].column)
        mxl = max((len(str(c.value)) for c in col if c.value), default=0)
        ws.column_dimensions[cl].width = min(max(mxl + 3, mn), mx)

def _model_color(choice):
    return C_FEAT if choice == 'Feat-XGBoost' \
        else C_XGB if choice == 'XGBoost' \
        else C_ERROR

def _sh(cell, text, color):
    cell.value     = text
    cell.font      = Font(name='Arial', bold=True, color=C_WHITE, size=11)
    cell.fill      = _fill(color)
    cell.alignment = _c()

def _avg_acc(folds):
    v = [r['acc'] for r in folds if r['model_choice'] != 'Error']
    return np.mean(v) if v else 0.0

def _std_acc(folds):
    v = [r['acc'] for r in folds if r['model_choice'] != 'Error']
    return np.std(v) if v else 0.0

def _avg_mem(folds):
    return np.mean([r['mem_peak_mb'] for r in folds])

def _max_mem(folds):
    return np.max([r['mem_peak_mb'] for r in folds])

def _total_mem(folds):
    return np.sum([r['mem_peak_mb'] for r in folds])


# ─────────────────────────────────────────────────────────────────────────────
def save_benchmark_to_excel(output_path, dataset_name, all_results, data_levels):
    ts         = datetime.now().strftime('%Y-%m-%d_%H-%M')
    excel_path = os.path.join(output_path,
                              f'benchmark_{dataset_name}_{ts}.xlsx')
    wb = Workbook()
    wb.remove(wb.active)

    _build_overview(wb, dataset_name, all_results, data_levels)
    _build_charts(wb, dataset_name, all_results, data_levels)
    _build_detail_sheets(wb, all_results, data_levels)
    _build_feat_selection(wb, all_results, data_levels)

    wb.save(excel_path)
    return excel_path


# ── Sheet 1: Overview ─────────────────────────────────────────────────────────
def _build_overview(wb, dataset_name, all_results, data_levels):
    ws = wb.create_sheet('Overview')

    # Tiêu đề
    _title_row(ws,
               f'Benchmark — {dataset_name}  |  '
               f'{datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
               f'Baseline vs Full Optimization (M10)',
               15, C_TITLE)

    # Sub-header nhóm
    #        A     B      C       D      E       F        G       H       I        J       K       L        M    N     O
    # Cấp | %Data | Mẫu | AccB | StdB | TimeB | RamAvgB | RamMaxB | AccM10 | StdM10 | TimeM10 | RamAvgM10 | RamMaxM10 | Chênh | Kết quả
    ws.merge_cells('A2:C2');  _sh(ws['A2'], 'Cấp dữ liệu',               C_TITLE)
    ws.merge_cells('D2:H2');  _sh(ws['D2'], 'Baseline',                   C_BASE)
    ws.merge_cells('I2:M2');  _sh(ws['I2'], 'Full Optimization (M10)',    C_M10)
    ws.merge_cells('N2:O2');  _sh(ws['N2'], 'So sánh',                    C_HDR)
    ws.row_dimensions[2].height = 20

    headers = [
        # Cấp dữ liệu
        'Cấp', '% Data', 'Số mẫu',
        # Baseline
        'Acc', 'Std Acc', 'Time (s)', 'RAM Avg (MB)', 'RAM Max (MB)',
        # M10
        'Acc', 'Std Acc', 'Time (s)', 'RAM Avg (MB)', 'RAM Max (MB)',
        # So sánh
        'Chênh lệch Acc', 'Kết quả',
    ]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font, c.fill, c.alignment, c.border = _wb(), _fill(C_HDR), _c(), _b()
    ws.row_dimensions[3].height = 30

    for i, lvl_data in enumerate(data_levels):
        lvl = lvl_data['level']
        row = i + 4
        rc  = C_GREY if i % 2 == 0 else C_WHITE

        fb = all_results[lvl]['baseline']['fold_results']
        fm = all_results[lvl]['m10']['fold_results']

        acc_b  = _avg_acc(fb);  acc_m  = _avg_acc(fm)
        std_b  = _std_acc(fb);  std_m  = _std_acc(fm)
        t_b    = all_results[lvl]['baseline']['total_time']
        t_m    = all_results[lvl]['m10']['total_time']
        ramA_b = _avg_mem(fb);  ramA_m = _avg_mem(fm)
        ramX_b = _max_mem(fb);  ramX_m = _max_mem(fm)
        diff   = acc_m - acc_b

        if diff > 0.001:
            result, rc_cmp = '✅ M10 tốt hơn',     C_WIN
        elif diff < -0.001:
            result, rc_cmp = '❌ Base tốt hơn',     C_LOSE
        else:
            result, rc_cmp = '➖ Tương đương',       C_TIE

        values = [
            lvl, f'{lvl_data["pct"]:.1f}%', lvl_data['n_sample'],
            round(acc_b,  4), round(std_b,  4), round(t_b,    1),
            round(ramA_b, 1), round(ramX_b, 1),
            round(acc_m,  4), round(std_m,  4), round(t_m,    1),
            round(ramA_m, 1), round(ramX_m, 1),
            round(diff,   4), result,
        ]
        for col, val in enumerate(values, 1):
            c = ws.cell(row=row, column=col, value=val)
            # RAM columns → màu cam nhạt
            if col in [7, 8, 12, 13]:
                c.fill = _fill(C_RAM)
            elif col >= 14:
                c.fill = _fill(rc_cmp)
            else:
                c.fill = _fill(rc)
            c.alignment = _c()
            c.border    = _b()
            if col in [4, 5, 9, 10, 14]:
                c.number_format = '0.0000'
            elif col in [6, 7, 8, 11, 12, 13]:
                c.number_format = '0.0'

    # Color scale
    n = len(data_levels)
    for col_letter in ['D', 'I']:
        ws.conditional_formatting.add(
            f'{col_letter}4:{col_letter}{n+3}',
            ColorScaleRule(
                start_type='min',        start_color='FFC7CE',
                mid_type='percentile',   mid_value=50, mid_color='FFEB9C',
                end_type='max',          end_color='C6EFCE'
            )
        )
    # Color scale RAM — đỏ = dùng nhiều, xanh = dùng ít
    for col_letter in ['G', 'H', 'L', 'M']:
        ws.conditional_formatting.add(
            f'{col_letter}4:{col_letter}{n+3}',
            ColorScaleRule(
                start_type='min', start_color='C6EFCE',
                end_type='max',   end_color='FFC7CE'
            )
        )

    # Tổng kết
    m10_wins  = sum(1 for ld in data_levels
                    if _avg_acc(all_results[ld['level']]['m10']['fold_results']) >
                       _avg_acc(all_results[ld['level']]['baseline']['fold_results']) + 0.001)
    base_wins = sum(1 for ld in data_levels
                    if _avg_acc(all_results[ld['level']]['baseline']['fold_results']) >
                       _avg_acc(all_results[ld['level']]['m10']['fold_results']) + 0.001)
    ties = n - m10_wins - base_wins

    # Tổng RAM
    total_ram_b = sum(_total_mem(all_results[ld['level']]['baseline']['fold_results'])
                      for ld in data_levels)
    total_ram_m = sum(_total_mem(all_results[ld['level']]['m10']['fold_results'])
                      for ld in data_levels)

    sum_row = n + 5
    ws.merge_cells(f'A{sum_row}:O{sum_row}')
    ws[f'A{sum_row}'].value = (
        f'📊 Tổng kết {n} cấp:  '
        f'✅ M10 tốt hơn = {m10_wins} cấp  |  '
        f'❌ Base tốt hơn = {base_wins} cấp  |  '
        f'➖ Tương đương = {ties} cấp  |  '
        f'💾 Tổng RAM Base = {total_ram_b:.0f} MB  |  '
        f'💾 Tổng RAM M10 = {total_ram_m:.0f} MB'
    )
    ws[f'A{sum_row}'].font      = Font(bold=True, size=11)
    ws[f'A{sum_row}'].alignment = _c()
    ws[f'A{sum_row}'].fill      = _fill(C_AVG)

    _auto_w(ws)


# ── Sheet 2: Biểu đồ ─────────────────────────────────────────────────────────
def _build_charts(wb, dataset_name, all_results, data_levels):
    ws = wb.create_sheet('Charts')
    _title_row(ws,
               f'Biểu đồ so sánh Baseline vs M10 — {dataset_name}',
               10, C_TITLE)

    # Bảng dữ liệu nguồn
    hdrs = [
        'Cấp', '% Data',
        'Acc Baseline', 'Acc M10',
        'Time Baseline (s)', 'Time M10 (s)',
        'RAM Avg Baseline (MB)', 'RAM Avg M10 (MB)',
        'RAM Max Baseline (MB)', 'RAM Max M10 (MB)',
    ]
    for col, h in enumerate(hdrs, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font, c.fill, c.alignment, c.border = _wb(), _fill(C_HDR), _c(), _b()

    for i, lvl_data in enumerate(data_levels):
        lvl = lvl_data['level']
        row = i + 4
        fb  = all_results[lvl]['baseline']['fold_results']
        fm  = all_results[lvl]['m10']['fold_results']
        vals = [
            lvl,
            f'{lvl_data["pct"]:.0f}%',
            round(_avg_acc(fb),  4),
            round(_avg_acc(fm),  4),
            round(all_results[lvl]['baseline']['total_time'], 1),
            round(all_results[lvl]['m10']['total_time'],      1),
            round(_avg_mem(fb),  1),
            round(_avg_mem(fm),  1),
            round(_max_mem(fb),  1),
            round(_max_mem(fm),  1),
        ]
        for col, val in enumerate(vals, 1):
            c = ws.cell(row=row, column=col, value=val)
            c.alignment, c.border = _c(), _b()
            if col in [7, 8, 9, 10]:
                c.fill = _fill(C_RAM)

    n          = len(data_levels)
    data_start = 4
    data_end   = data_start + n - 1
    cats = Reference(ws, min_col=2, min_row=data_start, max_row=data_end)

    def _make_chart(title, y_title, height=14, width=22):
        ch = LineChart()
        ch.title           = title
        ch.style           = 10
        ch.height          = height
        ch.width           = width
        ch.y_axis.title    = y_title
        ch.x_axis.title    = '% Data'
        return ch

    # Chart 1: Accuracy
    c1 = _make_chart('Accuracy theo cấp dữ liệu', 'Accuracy')
    c1.add_data(Reference(ws, min_col=3, min_row=3, max_row=data_end),
                titles_from_data=True)
    c1.add_data(Reference(ws, min_col=4, min_row=3, max_row=data_end),
                titles_from_data=True)
    c1.set_categories(cats)
    c1.series[0].marker.symbol = 'circle'
    c1.series[1].marker.symbol = 'diamond'
    ws.add_chart(c1, 'A' + str(data_end + 3))

    # Chart 2: Thời gian
    c2 = _make_chart('Thời gian chạy (giây)', 'Giây (s)')
    c2.add_data(Reference(ws, min_col=5, min_row=3, max_row=data_end),
                titles_from_data=True)
    c2.add_data(Reference(ws, min_col=6, min_row=3, max_row=data_end),
                titles_from_data=True)
    c2.set_categories(cats)
    c2.series[0].marker.symbol = 'circle'
    c2.series[1].marker.symbol = 'diamond'
    ws.add_chart(c2, 'L' + str(data_end + 3))

    # Chart 3: RAM Avg
    c3 = _make_chart('RAM trung bình (MB)', 'MB')
    c3.add_data(Reference(ws, min_col=7, min_row=3, max_row=data_end),
                titles_from_data=True)
    c3.add_data(Reference(ws, min_col=8, min_row=3, max_row=data_end),
                titles_from_data=True)
    c3.set_categories(cats)
    c3.series[0].marker.symbol = 'circle'
    c3.series[1].marker.symbol = 'diamond'
    ws.add_chart(c3, 'A' + str(data_end + 24))

    # Chart 4: RAM Max
    c4 = _make_chart('RAM peak tối đa (MB)', 'MB')
    c4.add_data(Reference(ws, min_col=9, min_row=3, max_row=data_end),
                titles_from_data=True)
    c4.add_data(Reference(ws, min_col=10, min_row=3, max_row=data_end),
                titles_from_data=True)
    c4.set_categories(cats)
    c4.series[0].marker.symbol = 'circle'
    c4.series[1].marker.symbol = 'diamond'
    ws.add_chart(c4, 'L' + str(data_end + 24))

    _auto_w(ws)


# ── Sheet 3: Chi tiết từng cấp ───────────────────────────────────────────────
def _build_detail_sheets(wb, all_results, data_levels):
    for lvl_data in data_levels:
        lvl = lvl_data['level']
        ws  = wb.create_sheet(f'Cap_{lvl}_{lvl_data["pct"]:.0f}pct')

        _title_row(ws,
                   f'Cấp {lvl} — {lvl_data["pct"]:.1f}% '
                   f'({lvl_data["n_sample"]}/{lvl_data["total_train"]} mẫu | '
                   f'{lvl_data["n_classes"]} class)',
                   14, C_TITLE)

        ws.merge_cells('A2:N2')
        ws['A2'].value     = '🟢 Feat-XGBoost    🟡 XGBoost gốc    🔴 Lỗi'
        ws['A2'].font      = Font(italic=True, size=9)
        ws['A2'].alignment = _c()

        headers = [
            'Phương án', 'Fold', 'Model chọn',
            'Val Acc Feat', 'Val Acc XGB',
            'Test Acc', 'F1 Macro', 'F1 Weight',
            'Precision', 'Recall',
            'Early Stop vòng',
            'RAM Avg (MB)', 'RAM Peak (MB)',   # ← thêm 2 cột RAM
            'Time (s)',
        ]
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=3, column=col, value=h)
            c.font, c.fill, c.alignment, c.border = _wb(), _fill(C_HDR), _c(), _b()

        current_row = 4
        for method_key in ['baseline', 'm10']:
            res    = all_results[lvl][method_key]
            folds  = res['fold_results']
            m_name = METHOD_NAMES[method_key]

            for fi, fold in enumerate(folds):
                rc = _model_color(fold['model_choice'])
                values = [
                    m_name if fi == 0 else '',
                    f'Fold {fi+1}',
                    fold['model_choice'],
                    fold['feat_acc_optuna'],
                    fold['default_acc_optuna'],
                    fold['acc'],
                    fold['f1_macro'],
                    fold['f1_weight'],
                    fold['precision'],
                    fold['recall'],
                    fold['early_stop_at'],
                    round(fold['mem_peak_mb'], 1),   # RAM avg (per fold)
                    round(fold['mem_peak_mb'], 1),   # RAM peak
                    round(fold['fold_time'], 2),
                ]
                for col, val in enumerate(values, 1):
                    c = ws.cell(row=current_row, column=col, value=val)
                    c.fill      = _fill(C_RAM if col in [12, 13] else rc)
                    c.alignment = _c()
                    c.border    = _b()
                    if col in [4, 5, 6, 7, 8, 9, 10]:
                        c.number_format = '0.0000'
                    elif col in [12, 13, 14]:
                        c.number_format = '0.0'
                current_row += 1

            # Average row
            n       = len(folds)
            avg_row = current_row
            m_fill  = _fill(C_BASE_L if method_key == 'baseline' else C_M10_L)

            ws.cell(row=avg_row, column=1,
                    value=f'Avg {m_name}').font = Font(bold=True)
            for col in [1, 2, 3, 11]:
                c = ws.cell(row=avg_row, column=col)
                c.fill, c.alignment, c.border = m_fill, _c(), _b()

            for col in range(4, 15):
                cl = get_column_letter(col)
                c  = ws.cell(row=avg_row, column=col)
                c.value     = f'=AVERAGE({cl}{avg_row-n}:{cl}{avg_row-1})'
                c.fill      = _fill(C_RAM) if col in [12, 13] else m_fill
                c.alignment = _c()
                c.border    = _b()
                c.font      = Font(bold=True)
                if col <= 10:
                    c.number_format = '0.0000'
                else:
                    c.number_format = '0.0'
            current_row += 2

        _auto_w(ws)


# ── Sheet 4: Feat Selection ──────────────────────────────────────────────────
def _build_feat_selection(wb, all_results, data_levels):
    ws = wb.create_sheet('Feat Selection')
    _title_row(ws, 'Thống kê lựa chọn Model + RAM — Baseline vs M10', 11, C_M10)

    headers = [
        'Cấp', '% Data', 'Số mẫu',
        'Feat Base (%)', 'XGB Base (%)',
        'Feat M10 (%)',  'XGB M10 (%)',
        'Early Stop M10',
        'RAM Avg Base (MB)', 'RAM Avg M10 (MB)',
        'RAM tiết kiệm (MB)',
    ]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font, c.fill, c.alignment, c.border = _wb(), _fill(C_HDR), _c(), _b()

    for i, lvl_data in enumerate(data_levels):
        lvl = lvl_data['level']
        row = i + 3
        rc  = C_GREY if i % 2 == 0 else C_WHITE

        fb = all_results[lvl]['baseline']['fold_results']
        fm = all_results[lvl]['m10']['fold_results']
        n  = len(fm)

        feat_b = sum(1 for r in fb if r['model_choice']=='Feat-XGBoost')/len(fb)*100
        xgb_b  = sum(1 for r in fb if r['model_choice']=='XGBoost')/len(fb)*100
        feat_m = sum(1 for r in fm if r['model_choice']=='Feat-XGBoost')/n*100
        xgb_m  = sum(1 for r in fm if r['model_choice']=='XGBoost')/n*100
        avg_es = np.mean([r['early_stop_at'] for r in fm])
        ram_b  = _avg_mem(fb)
        ram_m  = _avg_mem(fm)
        saved  = ram_b - ram_m   # dương = M10 dùng ít RAM hơn

        values = [
            lvl, f'{lvl_data["pct"]:.1f}%', lvl_data['n_sample'],
            f'{feat_b:.0f}%', f'{xgb_b:.0f}%',
            f'{feat_m:.0f}%', f'{xgb_m:.0f}%',
            round(avg_es, 0),
            round(ram_b, 1), round(ram_m, 1),
            round(saved, 1),
        ]
        for col, val in enumerate(values, 1):
            if col in [9, 10, 11]:
                fill_c = C_RAM
            elif col == 11 and saved > 0:
                fill_c = C_WIN
            elif col == 11 and saved < 0:
                fill_c = C_LOSE
            else:
                fill_c = rc
            c = ws.cell(row=row, column=col, value=val)
            c.fill, c.alignment, c.border = _fill(fill_c), _c(), _b()
            if col in [9, 10, 11]:
                c.number_format = '0.0'

    _auto_w(ws)