import sys

def check_package(package_name):
    try:
        pkg = __import__(package_name)
        version = getattr(pkg, '__version__', 'Không rõ phiên bản')
        print(f"[OK] Đã cài đặt {package_name} - Phiên bản: {version}")
    except ImportError:
        print(f"[LỖI] Chưa cài đặt {package_name}")

print(f"Phiên bản Python hiện tại: {sys.version}")
print("-" * 30)

packages = [
    'xgboost',
    'optuna',
    'torch',
    'sklearn', # scikit-learn
    'pandas',
    'numpy',
    'yaml',
    'joblib'
]

for p in packages:
    check_package(p)