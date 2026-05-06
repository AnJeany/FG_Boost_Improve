"""
BENCHMARK_LEVELS — 10 mức kiểm tra hiệu năng

Mỗi mức bật/tắt các tối ưu khác nhau để so sánh:
  - use_pruning      : Optuna MedianPruner (tối ưu #1)
  - use_early_stop   : Early stopping boosting (tối ưu #3)
  - use_parallel_std : Chạy song song 2 study (tối ưu #4)
  - n_trials         : Số lần Optuna thử
  - n_boost          : Số vòng boosting
  - patience         : Số vòng không cải thiện trước khi dừng
"""

BENCHMARK_LEVELS = {
    1: {
        'name':             'Baseline',
        'description':      'Không có tối ưu nào — chạy như gốc',
        'use_pruning':      False,
        'use_early_stop':   False,
        'use_parallel_std': False,
        'n_trials':         100,
        'n_boost':          100,
        'patience':         100,   # patience lớn = không early stop
    },
    2: {
        'name':             'Pruning Only',
        'description':      'Chỉ dùng Optuna MedianPruner',
        'use_pruning':      True,
        'use_early_stop':   False,
        'use_parallel_std': False,
        'n_trials':         100,
        'n_boost':          100,
        'patience':         100,
    },
    3: {
        'name':             'Early Stop Only',
        'description':      'Chỉ dùng Early stopping boosting',
        'use_pruning':      False,
        'use_early_stop':   True,
        'use_parallel_std': False,
        'n_trials':         100,
        'n_boost':          100,
        'patience':         10,
    },
    4: {
        'name':             'Pruning + Early Stop',
        'description':      'Pruning + Early stopping',
        'use_pruning':      True,
        'use_early_stop':   True,
        'use_parallel_std': False,
        'n_trials':         100,
        'n_boost':          100,
        'patience':         10,
    },
    5: {
        'name':             'Reduced Trials',
        'description':      'Giảm số trial Optuna xuống 50',
        'use_pruning':      False,
        'use_early_stop':   False,
        'use_parallel_std': False,
        'n_trials':         50,
        'n_boost':          100,
        'patience':         100,
    },
    6: {
        'name':             'Reduced Trials + Pruning',
        'description':      'Giảm trial + Pruning',
        'use_pruning':      True,
        'use_early_stop':   False,
        'use_parallel_std': False,
        'n_trials':         50,
        'n_boost':          100,
        'patience':         100,
    },
    7: {
        'name':             'Reduced Trials + Early Stop',
        'description':      'Giảm trial + Early stopping',
        'use_pruning':      False,
        'use_early_stop':   True,
        'use_parallel_std': False,
        'n_trials':         50,
        'n_boost':          100,
        'patience':         10,
    },
    8: {
        'name':             'Reduced Trials + Pruning + Early Stop',
        'description':      'Giảm trial + Pruning + Early stopping',
        'use_pruning':      True,
        'use_early_stop':   True,
        'use_parallel_std': False,
        'n_trials':         50,
        'n_boost':          100,
        'patience':         10,
    },
    9: {
        'name':             'Parallel Studies',
        'description':      'Chạy song song 2 study Optuna',
        'use_pruning':      False,
        'use_early_stop':   False,
        'use_parallel_std': True,
        'n_trials':         100,
        'n_boost':          100,
        'patience':         100,
    },
    10: {
        'name':             'Full Optimization',
        'description':      'Tất cả tối ưu: Pruning + Early stop + Parallel + Reduced trials',
        'use_pruning':      True,
        'use_early_stop':   True,
        'use_parallel_std': True,
        'n_trials':         50,
        'n_boost':          100,
        'patience':         10,
    },
}