# 快速开始指南

## 🎯 三种推理方式

### 方式1️⃣: PyQt5 图形界面（推荐）
最友好的使用方式，支持预览、实时反馈

```bash
python inference_app.py
```

或使用命令行参数指定配置：
```bash
python run_inference.py --model Vit_B_16 --size 224 --folder Inference
```

**功能特性：**
- 📷 实时图片预览
- 📊 可视化结果显示
- 📈 批量推理支持
- 💾 结果导出（CSV/TXT）
- 📉 统计信息展示

### 方式2️⃣: 命令行工具（快速推理）
适合脚本集成和批处理

```bash
# 推理单张图片
python cli_inference.py image.jpg --model Vit_B_16

# 批量推理文件夹
python cli_inference.py ./Inference --model Vit_B_16 --output result.json

# 指定自定义模型路径
python cli_inference.py image.jpg --checkpoint ./my_model.pth
```

**功能特性：**
- ⚡ 快速推理
- 📝 详细的控制台输出
- 💾 JSON格式导出
- 🔄 批量处理支持

### 方式3️⃣: Python API（编程调用）
集成到您的项目中

```python
from src.inference.inference_engine import InferenceEngine

# 初始化引擎
engine = InferenceEngine(
    model_name='Vit_B_16',
    checkpoint='experiments/Vit_B_16/checkpoints/best_Vit_B_16_model.pth',
    IMAGE_SIZE=(224, 224)
)

# 推理单张图片
result = engine.infer('Inference/image.jpg')
print(f"预测: {result['class']}, 置信度: {result['confidence']:.2%}")

# 批量推理
results = engine.batch_infer(['img1.jpg', 'img2.jpg'])
```

---

## 📦 安装依赖

### 一键安装（推荐）
```bash
pip install -r requirements_inference.txt
```

### 手动安装
```bash
pip install PyQt5>=5.15.0 opencv-python>=4.5.0 albumentations>=1.0.0
```

---

## 📁 使用步骤

### Step 1: 准备文件夹
```bash
# Inference文件夹会自动创建，放入要推理的图片即可
Inference/
├── product_001.jpg
├── product_002.png
└── ...
```

### Step 2: 运行应用
```bash
python inference_app.py
```

### Step 3: 初始化模型
1. 在UI中选择模型（Vit_B_16 / resnet50）
2. 点击"初始化模型"

### Step 4: 执行推理
- **单张**: 选择图片 → "推理选中图片"
- **批量**: 点击"批量推理所有图片"

### Step 5: 查看和导出结果
- 在"批量推理结果"选项卡查看结果表格
- 点击"导出结果为CSV"导出数据

---

## 🎨 UI界面布局

```
┌─────────────────────────────────────────────────────────────┐
│  配置区域 [选择模型] [图像大小] [初始化模型] [刷新] [状态]  │
├────────────────────┬────────────────────────────────────────┤
│  选项卡1: 图片推理  │  选项卡2: 批量结果  │  选项卡3: 统计  │
├────────────────────┼────────────────────────────────────────┤
│  图片列表          │  📷 图片预览 (400x400)                 │
│  ┌──────────────┐  │  ┌────────────────────────────────┐   │
│  │ image1.jpg   │  │  │                                │   │
│  │ image2.jpg   │  │  │                                │   │
│  │ image3.jpg   │  │  │                                │   │
│  │ ...          │  │  └────────────────────────────────┘   │
│  └──────────────┘  │                                        │
│                    │  📊 推理结果                            │
│  [推理选中]        │  ┌────────────────────────────────┐   │
│  [批量推理]        │  │ 预测类别: OK                   │   │
│  [打开文件夹]      │  │ 置信度: 95.23%                 │   │
│                    │  │ NG: 4.77%                      │   │
│                    │  │ OK: 95.23%                     │   │
│                    │  └────────────────────────────────┘   │
│                    │  [保存结果]                             │
└────────────────────┴────────────────────────────────────────┘
```

---

## 💡 常用命令

### PyQt5 UI
```bash
# 默认启动
python inference_app.py

# 指定模型启动
python run_inference.py --model resnet50

# 自定义图像大小和推理文件夹
python run_inference.py --size 256 --folder my_images/
```

### 命令行工具
```bash
# 推理单张图片并显示详细结果
python cli_inference.py test.jpg

# 批量推理并导出JSON
python cli_inference.py Inference/ --output results.json

# 使用自定义模型
python cli_inference.py images/ --checkpoint custom_model.pth

# 完整示例
python cli_inference.py ./Inference \
    --model Vit_B_16 \
    --size 224 \
    --output batch_results.json
```

### Python脚本集成
```python
from src.inference.inference_engine import InferenceEngine

# 创建引擎
engine = InferenceEngine('Vit_B_16', 'experiments/Vit_B_16/checkpoints/best_Vit_B_16_model.pth')

# 推理
result = engine.infer('image.jpg')
print(f"✅ {result['class']}: {result['confidence']:.2%}")
```

---

## 📊 输出格式

### 单张推理（文本格式）
```
推理结果:
==================================================
预测类别: OK
置信度: 95.23%

各类别概率:
  NG: 4.77%
  OK: 95.23%
```

### 批量推理（CSV格式）
```csv
图片名称,预测类别,置信度,推理时间
product_001.jpg,OK,0.9523
product_002.jpg,NG,0.8765
product_003.jpg,OK,0.9891
```

### 批量推理（JSON格式）
```json
[
  {
    "image_path": "image1.jpg",
    "class": "OK",
    "confidence": 0.9523,
    "probabilities": {
      "NG": 0.0477,
      "OK": 0.9523
    },
    "timestamp": "2024-03-02T10:30:45"
  }
]
```

---

## ⚙️ 配置说明

### inference_engine.py
核心推理引擎，支持：
- 自动GPU检测和使用
- 多模型支持
- 图像预处理标准化
- 单张和批量推理

### inference_ui.py
PyQt5用户界面，包含：
- 多选项卡设计
- 实时图片预览
- 后台推理线程
- 结果表格展示
- 导出功能

---

## 🐛 故障排除

| 问题 | 解决方案 |
|------|--------|
| 模块导入错误 | 确保在项目根目录运行脚本 |
| PyQt5找不到 | 运行 `pip install PyQt5` |
| 模型文件找不到 | 检查 `experiments/` 文件夹和模型名称 |
| 推理很慢 | 检查GPU驱动，减小图像大小 |
| 图片预览显示不了 | 检查图片格式是否支持（jpg/png/bmp） |

---

## 📝 文件映射

| 文件 | 功能 |
|------|------|
| `inference_app.py` | PyQt5应用入口 |
| `run_inference.py` | 参数化启动脚本 |
| `cli_inference.py` | 命令行工具 |
| `src/inference/inference_engine.py` | 推理引擎核心 |
| `ui/inference_ui.py` | UI界面实现 |
| `Inference/` | 推理图片输入文件夹 |
| `Inference/results/` | 推理结果输出文件夹 |

---

## 🚀 性能优化建议

1. **GPU加速**: 确保CUDA已安装和配置
2. **批量处理**: 使用批量推理而非逐张推理
3. **图像预处理**: 使用合适的图像大小（224适合大多数模型）
4. **模型选择**: Vit_B_16通常精度更高，resnet50更快

---

## 📞 技术支持

遇到问题时检查：
1. ✅ Python版本 ≥ 3.7
2. ✅ PyTorch正确安装
3. ✅ 模型文件完整
4. ✅ 图片格式支持
5. ✅ 路径正确无误

---

**祝您使用愉快！** 🎉
