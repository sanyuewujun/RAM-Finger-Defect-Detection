# PyQt5 推理系统完整实现指南

## 📋 项目概述

这是一个完整的深度学习模型推理系统，包含：
- ✅ **PyQt5 GUI应用** - 专业级用户界面
- ✅ **命令行工具** - 自动化批处理
- ✅ **Python API** - 灵活的编程接口
- ✅ **后台推理线程** - 非阻塞式推理
- ✅ **结果导出功能** - CSV/JSON/TXT格式

---

## 🎯 核心功能模块

### 1. 推理引擎 (inference_engine.py)
**职责**: 模型加载和图片推理

```
InferenceEngine
├── __init__()          # 初始化模型和预处理器
├── _init_model()       # 加载模型和权重
├── _init_transform()   # 初始化图像变换
├── infer()            # 单张图片推理
└── batch_infer()      # 批量推理
```

**关键特性**:
- 自动GPU检测
- 标准化图像预处理 (ImageNet标准)
- 返回置信度和类别概率
- 错误处理和日志

### 2. PyQt5 用户界面 (inference_ui.py)
**职责**: 提供专业的图形交互界面

```
InferenceUI (QMainWindow)
├── 配置区 (QGroupBox)
│   ├── 模型选择
│   ├── 图像大小设置
│   ├── 文件夹管理
│   └── 模型初始化
├── 选项卡1: 图片推理
│   ├── 图片列表 (QListWidget)
│   ├── 图片预览 (QLabel)
│   └── 推理结果显示 (QTextEdit)
├── 选项卡2: 批量结果
│   └── 结果表格 (QTableWidget)
└── 选项卡3: 统计信息
    └── 统计文本 (QTextEdit)

InferenceWorker (QThread)
├── 后台推理线程
├── 进度信号
└── 结果回调
```

**关键特性**:
- 多选项卡设计
- 实时图片预览
- 非阻塞式后台推理
- 可视化结果展示
- 拖拽式文件管理

### 3. 命令行工具 (cli_inference.py)
**职责**: 快速命令行推理

```
cli_inference.py
├── 单张图片推理
├── 批量文件夹推理
├── 结果JSON导出
├── 详细控制台输出
└── 统计信息汇总
```

**关键特性**:
- 灵活的命令行参数
- 实时进度显示
- 美化的控制台输出
- JSON格式导出

---

## 📁 项目结构

```
DeepLearning/
│
├── src/
│   ├── inference/                    # 推理模块 (新增)
│   │   ├── __init__.py
│   │   └── inference_engine.py       # 推理引擎核心
│   ├── models/
│   │   └── models.py                 # 模型定义
│   ├── training/
│   │   └── trainer.py                # 训练代码
│   ├── data_loader/
│   │   └── dataset.py                # 数据加载
│   └── utils/
│       └── metrics.py                # 评估指标
│
├── ui/                               # UI模块 (新增)
│   ├── __init__.py
│   └── inference_ui.py               # PyQt5界面
│
├── Inference/                        # 推理文件夹 (新增)
│   ├── image1.jpg                    # 放入要推理的图片
│   ├── image2.jpg
│   └── results/                      # 推理结果输出
│       ├── results.csv
│       └── results.json
│
├── experiments/
│   └── Vit_B_16/
│       └── checkpoints/
│           └── best_Vit_B_16_model.pth
│
├── inference_app.py                  # PyQt5应用入口 (新增)
├── run_inference.py                  # 参数化启动脚本 (新增)
├── cli_inference.py                  # 命令行工具 (新增)
├── main.py                           # 原有的主脚本
├── requirements.txt                  # 原有依赖
├── requirements_inference.txt        # 推理依赖 (新增)
├── INFERENCE_README.md               # 详细文档 (新增)
├── QUICK_START.md                    # 快速开始 (新增)
└── README.md                         # 项目README
```

---

## 🚀 三种使用方式详解

### 方式1: PyQt5 图形界面
**最友好，推荐新手**

```bash
# 启动应用
python inference_app.py

# 或使用参数化启动
python run_inference.py --model Vit_B_16 --size 224
```

**优势**:
- 📷 实时图片预览
- 🖱️ 鼠标点击操作
- 📊 可视化结果
- 💾 一键导出

**劣势**:
- 需要图形界面环境
- 单次推理较慢

### 方式2: 命令行工具
**最快捷，适合自动化**

```bash
# 推理单张图片
python cli_inference.py image.jpg

# 批量推理文件夹
python cli_inference.py ./Inference --output results.json
```

**优势**:
- ⚡ 快速推理
- 🔄 支持批处理
- 📝 可集成脚本
- 🎯 精确控制

**劣势**:
- 需要命令行知识
- 无实时预览

### 方式3: Python API
**最灵活，适合集成**

```python
from src.inference.inference_engine import InferenceEngine

# 初始化
engine = InferenceEngine('Vit_B_16', 'checkpoint.pth')

# 推理
result = engine.infer('image.jpg')
print(f"预测: {result['class']}")
```

**优势**:
- 🔧 完全可定制
- 🔗 易于集成
- 🎯 精确控制
- 📈 便于扩展

**劣势**:
- 需要编程知识
- 代码量较大

---

## 💻 安装与配置

### 步骤1: 安装依赖
```bash
# 安装推理依赖
pip install -r requirements_inference.txt

# 或个别安装
pip install PyQt5 opencv-python albumentations torch torchvision
```

### 步骤2: 验证环境
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import PyQt5; print('PyQt5: OK')"
python -c "import cv2; print('OpenCV: OK')"
```

### 步骤3: 准备数据
```bash
# 创建推理文件夹
mkdir Inference
# 复制图片到 Inference/ 文件夹
cp your_images/*.jpg Inference/
```

---

## 🔄 工作流程

```
用户输入图片
    ↓
[Inference文件夹]
    ↓
推理引擎处理
  ├─ 读取图片
  ├─ 图像预处理
  ├─ 模型推理
  └─ 后处理结果
    ↓
结果展示
  ├─ UI显示
  ├─ 控制台输出
  └─ 文件保存
    ↓
导出结果 (CSV/JSON/TXT)
```

---

## 📊 推理结果说明

### 返回值结构
```python
{
    'class': 'OK',              # 预测类别
    'class_id': 1,              # 类别ID
    'confidence': 0.9523,       # 置信度
    'probabilities': {          # 所有类别概率
        'NG': 0.0477,
        'OK': 0.9523
    },
    'class_names': ['NG', 'OK'] # 类别列表
}
```

### 错误处理
```python
{
    'image_path': 'image.jpg',
    'error': '无法读取图片: ...'
}
```

---

## 🎯 常见使用场景

### 场景1: 工业质检
```bash
# 批量检测产品缺陷
python cli_inference.py products/ --model Vit_B_16 --output quality_check.csv
```

### 场景2: 实时监控
```python
from src.inference.inference_engine import InferenceEngine
import os
from pathlib import Path

engine = InferenceEngine('Vit_B_16', 'best_model.pth')
watch_folder = Path('camera_input')

while True:
    for image in watch_folder.glob('*.jpg'):
        result = engine.infer(image)
        if result['class'] == 'NG':
            print(f"⚠️ 异常检测: {image.name}")
        image.unlink()  # 删除已处理的图片
```

### 场景3: 批量导入
```bash
# 导入一个月的数据进行分析
python cli_inference.py data/2024-03 --output monthly_report.json
```

### 场景4: Web服务集成
```python
from flask import Flask, request
from src.inference.inference_engine import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine('Vit_B_16', 'best_model.pth')

@app.route('/infer', methods=['POST'])
def infer():
    file = request.files['image']
    file.save('temp.jpg')
    result = engine.infer('temp.jpg')
    return result
```

---

## 🐛 常见问题排查

| 错误 | 原因 | 解决 |
|------|------|------|
| ModuleNotFoundError: PyQt5 | 未安装PyQt5 | `pip install PyQt5` |
| CUDA out of memory | GPU内存不足 | 减小batch_size或image_size |
| 模型文件找不到 | 路径错误 | 检查experiments/文件夹 |
| 图片读取失败 | 格式不支持 | 使用jpg/png/bmp格式 |
| 推理结果异常 | 预处理不匹配 | 检查IMAGE_SIZE设置 |

---

## 🔧 进阶配置

### 自定义模型支持
在 `src/models/models.py` 中添加新模型：

```python
def select_model(self, model_name):
    if model_name == "custom_model":
        return self.custom_model()
```

### 自定义预处理
在 `inference_engine.py` 中修改 `_init_transform()`:

```python
self.transform = A.Compose([
    # 自定义变换
    A.Resize(224, 224),
    A.CustomAugmentation(),
])
```

### 自定义输出格式
修改 `cli_inference.py` 中的输出部分：

```python
def print_result(result):
    # 自定义输出格式
    pass
```

---

## 📈 性能基准

| 模型 | 速度 | 精度 | GPU内存 |
|------|------|------|---------|
| ResNet50 | ⚡⚡⚡ | ⭐⭐⭐ | 4GB |
| Vit_B_16 | ⚡⚡ | ⭐⭐⭐⭐ | 8GB |

---

## 📚 文件清单

### 新增文件 (推理系统)
- [x] `src/inference/inference_engine.py` - 推理引擎
- [x] `src/inference/__init__.py` - 包初始化
- [x] `ui/inference_ui.py` - UI界面
- [x] `ui/__init__.py` - 包初始化
- [x] `inference_app.py` - 应用入口
- [x] `run_inference.py` - 启动脚本
- [x] `cli_inference.py` - 命令行工具
- [x] `requirements_inference.txt` - 依赖文件
- [x] `INFERENCE_README.md` - 详细文档
- [x] `QUICK_START.md` - 快速开始
- [x] `Inference/` - 推理文件夹
- [x] `Inference/results/` - 结果输出文件夹

### 验证清单
- [x] 语法检查通过
- [x] 导入依赖检查
- [x] 文件夹结构完整
- [x] 文档齐全

---

## 🎓 学习路径

1. **入门**: 阅读 [QUICK_START.md](QUICK_START.md)
2. **基础**: 使用PyQt5界面体验功能
3. **进阶**: 学习命令行工具参数
4. **高级**: 阅读源码，集成API到自己的项目

---

## 📞 技术支持

遇到问题时：
1. 查看 [INFERENCE_README.md](INFERENCE_README.md) 的FAQ部分
2. 检查[QUICK_START.md](QUICK_START.md) 的故障排除表
3. 验证环境和依赖是否正确安装

---

## 🎉 开始使用

```bash
# 快速启动（推荐）
python inference_app.py

# 或者命令行
python cli_inference.py ./Inference --model Vit_B_16 --output result.json
```

**祝您使用愉快！** 🚀
