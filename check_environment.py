"""
环境检查脚本 - 验证推理系统所需的所有依赖
"""
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"✓ Python: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ❌ 需要Python 3.7及以上版本")
        return False
    return True

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: 未安装")
        return False

def check_files():
    """检查必要的文件和文件夹"""
    root = Path('.')
    
    required_files = [
        'src/inference/inference_engine.py',
        'ui/inference_ui.py',
        'inference_app.py',
        'cli_inference.py',
        'requirements_inference.txt',
    ]
    
    print("\n📁 文件检查:")
    all_exist = True
    for file_path in required_files:
        path = root / file_path
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}: 不存在")
            all_exist = False
    
    return all_exist

def check_folders():
    """检查文件夹"""
    print("\n📂 文件夹检查:")
    
    required_folders = [
        'src',
        'ui',
        'Inference',
        'experiments',
    ]
    
    root = Path('.')
    all_exist = True
    for folder in required_folders:
        path = root / folder
        if path.exists():
            print(f"✓ {folder}/")
        else:
            print(f"✗ {folder}/: 不存在")
            all_exist = False
    
    # 创建缺失的关键文件夹
    (root / 'Inference').mkdir(exist_ok=True)
    (root / 'Inference' / 'results').mkdir(exist_ok=True)
    
    return all_exist

def check_gpu():
    """检查GPU可用性"""
    print("\n🖥️  GPU检查:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: 可用")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print(f"⚠ CUDA: 不可用（将使用CPU推理）")
        return True
    except ImportError:
        print(f"✗ PyTorch: 未安装")
        return False

def main():
    print("=" * 60)
    print("🔍 推理系统环境检查")
    print("=" * 60)
    
    print("\n📦 环境检查:")
    checks = [
        ("Python", check_python_version()),
        ("PyTorch", check_package("torch")),
        ("PyQt5", check_package("PyQt5")),
        ("OpenCV", check_package("cv2", "cv2")),
        ("Albumentations", check_package("albumentations")),
        ("NumPy", check_package("numpy")),
    ]
    
    print("\n" + "=" * 60)
    print("🔗 项目文件检查:")
    files_ok = check_files()
    folders_ok = check_folders()
    
    print("\n" + "=" * 60)
    gpu_ok = check_gpu()
    
    print("\n" + "=" * 60)
    print("📊 检查结果汇总:")
    print("=" * 60)
    
    all_packages_ok = all(check[1] for check in checks)
    
    if all_packages_ok and files_ok and folders_ok:
        print("✅ 所有检查通过！系统可以正常使用")
        print("\n🚀 快速开始:")
        print("  1. 将图片放入 Inference/ 文件夹")
        print("  2. 运行: python inference_app.py")
        print("  3. 选择模型 → 初始化 → 推理")
        return 0
    else:
        print("⚠️  存在问题需要修复:")
        
        missing_packages = [check[0] for check in checks if not check[1]]
        if missing_packages:
            print(f"\n缺失的包: {', '.join(missing_packages)}")
            print("运行: pip install -r requirements_inference.txt")
        
        if not files_ok:
            print("\n缺失的文件，请检查项目完整性")
        
        if not folders_ok:
            print("\n缺失的文件夹，已自动创建 Inference 和 Inference/results")
        
        print("\n💡 建议:")
        print("  1. 确保在项目根目录运行此脚本")
        print("  2. 安装所有依赖: pip install -r requirements_inference.txt")
        print("  3. 重新运行此脚本确认")
        
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
