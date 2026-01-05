#!/usr/bin/env python3
"""检测当前环境的 Python / PyTorch / CUDA 信息并以 JSON 输出。"""
import sys
import platform
import subprocess
import json


def gather_info():
	info = {}
	info["python_version"] = sys.version.replace("\n", " ")
	info["platform"] = platform.platform()

	# PyTorch 信息
	try:
		import torch
		info["torch_installed"] = True
		info["torch_version"] = getattr(torch, "__version__", None)
		info["cuda_available"] = torch.cuda.is_available()
		info["cuda_version_torch"] = getattr(torch.version, "cuda", None)
		try:
			info["cudnn_version"] = torch.backends.cudnn.version()
		except Exception:
			info["cudnn_version"] = None

		if info["cuda_available"]:
			info["gpu_count"] = torch.cuda.device_count()
			gpus = []
			for i in range(info["gpu_count"]):
				try:
					name = torch.cuda.get_device_name(i)
				except Exception:
					name = None
				try:
					capability = torch.cuda.get_device_capability(i)
				except Exception:
					capability = None
				gpus.append({"id": i, "name": name, "capability": capability})
			info["gpus"] = gpus
		else:
			info["gpu_count"] = 0
			info["gpus"] = []
	except Exception as e:
		info["torch_installed"] = False
		info["torch_error"] = str(e)

	# 尝试检测 nvcc 版本（如果已安装 CUDA Toolkit）
	try:
		cp = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
		info["nvcc_version_raw"] = cp.stdout.strip()
	except Exception:
		info["nvcc_version_raw"] = None

	# 尝试检测 nvidia-smi 输出（如果有 NVIDIA 驱动）
	try:
		cp2 = subprocess.run([
			"nvidia-smi",
			"--query-gpu=name,driver_version,memory.total",
			"--format=csv,noheader"
		], capture_output=True, text=True, check=True)
		info["nvidia_smi"] = [l.strip() for l in cp2.stdout.strip().splitlines() if l.strip()]
	except Exception:
		info["nvidia_smi"] = None

	return info


def main():
	info = gather_info()
	print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
