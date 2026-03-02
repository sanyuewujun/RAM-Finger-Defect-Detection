# PyQt5 推理应用使用指南

## 功能概述

这是一个基于PyQt5的深度学习模型推理工具，支持：
- ✅ 单张图片推理
- ✅ 批量图片推理
- ✅ 实时图片预览
- ✅ 置信度展示
- ✅ 结果导出（CSV格式）
- ✅ 多模型支持（Vit_B_16, ResNet50）

## 项目结构

```
DeepLearning/
├── Inference/                    # 推理文件夹（放置要推理的图片）
│   └── results/                  # 推理结果保存文件夹
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inference_engine.py   # 推理引擎核心代码
│   ├── models/
│   ├── training/
│   ├── data_loader/
│   └── utils/
├── ui/
│   ├── __init__.py
│   └── inference_ui.py           # PyQt5 UI界面
├── inference_app.py              # 应用入口文件
├── requirements_inference.txt    # 推理应用依赖
└── main.py                       # 原有的训练/测试脚本
```

## 安装依赖

### 方法1：安装推理专用依赖
```bash
pip install -r requirements_inference.txt
```

### 方法2：使用conda（推荐）
```bash
conda activate GF  # 激活您的conda环境
pip install PyQt5>=5.15.0 opencv-python>=4.5.0
```

## 使用步骤

### 1. 准备图片
将要推理的图片放入 `Inference/` 文件夹（支持格式：jpg, jpeg, png, bmp, tiff）

```
Inference/
├── image1.jpg
├── image2.png
└── image3.jpg
```

### 2. 运行应用
```bash
python inference_app.py
```

或直接在VS Code中运行：
```bash
F5 或 Ctrl+F5
```

### 3. 初始化模型
1. 在UI中选择要使用的模型（Vit_B_16 或 resnet50）
2. 设置图像大小（默认224x224）
3. 点击"初始化模型"按钮

### 4. 推理图片

#### 单张图片推理：
1. 在左侧列表中选择一张图片
2. 点击"推理选中图片"
3. 查看右侧的预览和结果

#### 批量推理：
1. 点击"批量推理所有图片"
2. 等待进度条完成
3. 查看"批量推理结果"选项卡

### 5. 导出结果
- 单张结果：点击"保存结果"（保存为txt文件）
- 批量结果：切换到"批量推理结果"选项卡，点击"导出结果为CSV"

## UI界面详解

### 选项卡1：图片推理
- **左侧**：Inference文件夹中的图片列表
- **右侧上方**：选中图片的预览
- **右侧下方**：推理结果（预测类别、置信度、各类别概率）
- **按钮**：
  - 推理选中图片：对当前选中图片进行推理
  - 批量推理所有图片：对所有图片进行推理

### 选项卡2：批量推理结果
- **表格**：显示所有推理结果
  - 图片名称
  - 预测类别（OK为绿色，NG为红色）
  - 置信度
  - 查看按钮（预览该图片）
- **导出CSV**：将结果导出为CSV文件

### 选项卡3：统计信息
- 推理统计汇总
- 各类别数量和比例

## 配置区域

| 选项 | 说明 | 默认值 |
|------|------|--------|
| 选择模型 | Vit_B_16 或 resnet50 | Vit_B_16 |
| 图像大小 | 输入图像尺寸 | 224 |
| 推理文件夹 | 推理图片所在位置 | ./Inference |
| 刷新图片列表 | 重新加载Inference文件夹 | - |
| 初始化模型 | 加载模型权重 | - |

## 文件说明

### inference_engine.py
推理引擎核心代码，提供：
- `InferenceEngine` 类：模型加载和推理
- `infer()` 方法：单张图片推理
- `batch_infer()` 方法：批量推理

示例用法：
```python
from src.inference.inference_engine import InferenceEngine

# 初始化推理引擎
engine = InferenceEngine(
    model_name='Vit_B_16',
    checkpoint='experiments/Vit_B_16/checkpoints/best_Vit_B_16_model.pth',
    IMAGE_SIZE=(224, 224)
)

# 推理单张图片
result = engine.infer('Inference/image1.jpg')
print(f"预测类别: {result['class']}")
print(f"置信度: {result['confidence']:.2%}")

# 批量推理
results = engine.batch_infer(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### inference_ui.py
PyQt5用户界面代码，提供：
- `InferenceUI` 类：主窗口
- `InferenceWorker` 线程类：后台推理处理

## 常见问题

### Q1: 如何改变推理文件夹位置？
修改 `inference_ui.py` 中的 `__init__` 方法：
```python
self.inference_folder = Path("自定义路径")  # 修改此行
```

### Q2: 推理很慢，如何加速？
1. 确保GPU驱动正确安装
2. 检查CUDA是否可用：运行应用时查看状态是否显示使用GPU
3. 减少图像大小（如改为160x160）

### Q3: 如何支持更多的模型？
修改 `models.py` 和 `inference_engine.py`：
1. 在 `ModelSelection.select_model()` 中添加新模型
2. 在 `InferenceUI._create_config_layout()` 中的模型组合框添加选项

### Q4: 结果文件保存在哪里？
- 单张结果：`Inference/results/image_name_result_YYYYMMDD_HHMMSS.txt`
- 批量结果：`Inference/results/inference_results_YYYYMMDD_HHMMSS.csv`

### Q5: 如何处理推理错误？
- 检查图片格式是否支持
- 确保图片文件完整
- 查看错误消息中的详细信息

## 输出示例

### 单张推理结果
```
推理结果:
==================================================
预测类别: OK
置信度: 95.23%

各类别概率:
  NG: 4.77%
  OK: 95.23%
```

### CSV导出格式
```csv
图片名称,预测类别,置信度,推理时间
image1.jpg,OK,0.9523
image2.jpg,NG,0.8765
image3.jpg,OK,0.9891
```

## 技术栈

- **PyQt5**：UI框架
- **PyTorch**：深度学习框架
- **OpenCV**：图像处理
- **Albumentations**：图像增强和预处理
- **Python 3.7+**

## 注意事项

1. ⚠️ 首次运行需要初始化模型，时间较长
2. ⚠️ 推理图片应与模型训练时的图像大小相同（默认224x224）
3. ⚠️ 确保模型权重文件存在于 `experiments/` 文件夹中
4. ⚠️ Windows路径分隔符自动处理，不需要手动修改
5. ⚠️ 大批量推理时，进度条会实时更新

## 许可证

MIT License

## 问题反馈

如遇到问题，请检查：
1. 是否安装了所有依赖包
2. 模型文件路径是否正确
3. 输入图片格式是否支持
4. Python版本是否≥3.7
