import platform
import subprocess
import torch

def get_python_version():
    return platform.python_version()

def get_pytorch_version():
    return torch.__version__

def get_cuda_version():
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        for line in nvcc_output.split('\n'):
            if 'release' in line:
                return line.split('release')[-1].split(',')[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "未找到CUDA"

def get_cudnn_version():
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetDriverVersion(handle)
        # cuDNN版本通常不在这里显示，需要手动检查或查看安装路径下的版本信息文件
        return "请手动检查cuDNN版本"
    except ModuleNotFoundError:
        return "请安装nvidia-smi工具"
    except Exception as e:
        return f"获取cuDNN版本失败: {e}"

def main():
    print("Python版本:", get_python_version())
    print("PyTorch版本:", get_pytorch_version())
    print("CUDA版本:", get_cuda_version())
    print("cuDNN版本:", get_cudnn_version())

if __name__ == "__main__":
    main()
