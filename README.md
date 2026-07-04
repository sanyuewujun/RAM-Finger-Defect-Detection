# 基于深度学习的内存条金手指不良判断系统

一个用于自动检测内存条金手指不良品的深度学习图像分类系统，支持多种先进的神经网络架构，提供完整的训练、测试、推理和可视化功能。

## 📋 项目概述

### 背景与目标

内存条（RAM）金手指质量直接影响产品可靠性及后续装配效果。传统人工目视检验效率低、准确率波动大。本项目基于深度学习构建自动化、可视化的不良品识别系统，实现：

- **全自动化判定**：替代人工，实现金手指合格性自动归类（OK/NG）
- **生产线适配**：支持文件夹持续扫描与批量推理
- **工程化落地**：提供 PyQt5 GUI 交互界面，降低操作门槛
- **模型迭代**：内置完整的训练与测试评估闭环

## 🏗️ 项目结构

```
DeepLearning/
├── main.py                    # 主入口：训练/测试模式切换
├── inference_app.py          # GUI推理应用启动入口
├── cli_inference.py          # 命令行推理工具
├── system_introduction.md    # 系统介绍文档（答辩用）
│
├── src/                      # 核心源代码
│   ├── models/
│   │   └── models.py         # 模型定义（ResNeXt/Swin Transformer）
│   ├── training/
│   │   └── trainer.py        # 训练器（含早停、学习率调度）
│   ├── inference/
│   │   └── inference_engine.py  # 推理引擎
│   ├── data_loader/
│   │   └── dataset.py        # 数据加载器
│   ├── utils/
│   │   ├── metrics.py        # 评估指标计算（ROC/PR曲线）
│   │   └── result_manager.py # 结果管理与数据库操作
│   └── Environment.py        # 环境配置
│
├── ui/
│   └── inference_ui.py       # PyQt5图形界面
│
├── tests/
│   └── testmodel.py          # 模型测试模块
│
├── tools/
│   └── result_analyzer.py    # 结果分析工具
│
├── experiments/              # 实验结果存储
│   └── Swin_V2_B/           # 各模型实验目录
│       ├── checkpoints/      # 模型权重
│       ├── pltout/          # 训练曲线图
│       └── test_metrics/    # 测试结果
│
├── Inference/               # 推理输入目录
│   ├── OK/                 # 合格品样本
│   ├── NG/                 # 不良品样本
│   └── results/            # 推理结果存储
│
└── AIResult/               # 自动分类输出目录
    ├── OK/                 # 判定为合格的图片
    └── NG/                 # 判定为不良的图片
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- torchvision
- PyQt5
- OpenCV
- Albumentations
- scikit-learn
- matplotlib

### 安装依赖

```bash
pip install torch torchvision opencv-python albumentations PyQt5 scikit-learn matplotlib pandas pymysql
```

### 训练模型

```python
# main.py 配置
MODE = 'train'                          # 运行模式
cmodel_name = 'Swin_V2_B'               # 选择模型
num_epochs = 100                        # 训练轮次
batch_size = 16                         # 批处理大小
IMAGE_SIZE = (224, 224)                 # 图像尺寸
Aug = 0                                 # 数据增强开关

# 运行训练
python main.py
```

### 测试模型

```python
# main.py 配置
MODE = 'test'
model_name = 'Swin_V2_B'
checkpoint = f'experiments/{model_name}/checkpoints/best_{model_name}_model.pth'

# 运行测试
python main.py
```

### 启动GUI推理

```bash
python inference_app.py
```

### 命令行推理

```bash
# 单张图片推理
python cli_inference.py image.jpg --model Swin_V2_B

# 批量推理
python cli_inference.py ./images/ --model ResNeXt50_32X4D

# 导出结果
python cli_inference.py image.jpg --output result.json
```

## 🧠 支持的模型架构

| 模型 | 类型 | 特点 |
|------|------|------|
| **ResNeXt50_32X4D** | CNN | 分组卷积，平衡精度与速度 |
| **ResNeXt101_32X8d** | CNN | 更深的网络，特征提取能力强 |
| **ResNeXt101_64X4D** | CNN | 更宽的网络，并行计算优化 |
| **Swin_B** | Transformer | 移动窗口机制，层次化特征 |
| **Swin_V2_B** | Transformer | Swin V2优化版本，训练更稳定 |

### 模型微调策略

所有模型均采用迁移学习策略：

1. **加载预训练权重**（ImageNet）
2. **冻结主干网络**参数
3. **解冻深层特征层**（layer4/最后一个Transformer Block）
4. **替换分类头**为自定义全连接层

```python
# 示例：Swin_V2_B 微调
model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False
# 解冻最后一个Block
for param in model.features[-1].parameters():
    param.requires_grad = True
# 替换分类头
model.head = nn.Sequential(
    nn.Linear(num_ftrs, 768),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(768, num_classes)
)
```

## ⚙️ 训练配置

### 优化器与学习率调度

```python
# 分层学习率设置
LR_CUSTOM_HEAD = 1e-4      # 分类头学习率
LR_LAST_BLOCK = 1e-5       # 特征层学习率

optimizer = AdamW([
    {'params': model.head.parameters(), 'lr': LR_CUSTOM_HEAD},
    {'params': model.features[-1].parameters(), 'lr': LR_LAST_BLOCK}
], weight_decay=1e-4)

# OneCycleLR动态学习率
scheduler = OneCycleLR(
    optimizer,
    max_lr=[LR_CUSTOM_HEAD, LR_LAST_BLOCK],
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    pct_start=0.1
)
```

### 早停机制

```python
# 双重早停策略
1. 损失阈值策略：验证损失 ≤ 0.0001 时停止
2. Patience策略：连续50个epoch无改善时停止
```

## 🖥️ GUI功能介绍

### 1. 配置区域

- **模型选择**：下拉框选择预训练模型
- **图像大小**：设置输入图像尺寸（默认224x224）
- **推理文件夹**：显示当前监控的输入目录
- **状态显示**：模型初始化状态指示

### 2. 图片推理选项卡

- **图片列表**：显示Inference文件夹中的所有图片
- **图片预览**：选中图片的实时预览
- **推理结果**：概率分布表格展示
- **操作按钮**：
  - 推理选中图片
  - 批量推理所有图片
  - 自动推理并移动（自动分类到OK/NG文件夹）
  - 持续扫描（监控文件夹新图片）

### 3. 统计信息选项卡

- 总处理数量
- 类别分布（OK/NG比例）
- 平均置信度
- 实时统计更新

### 4. 推理结果记录选项卡

- 表格展示所有推理记录
- 支持CSV导出
- 高亮显示预测结果
- 显示置信度、耗时、时间戳

### 5. 训练和测试选项卡

- **训练配置**：选择模型、设置轮次、批大小
- **测试配置**：选择模型、加载权重、设置批大小
- **日志输出**：实时显示训练/测试日志

## 📊 评估指标

系统计算并保存以下评估指标：

| 指标 | 说明 |
|------|------|
| **Accuracy** | 总体准确率 |
| **Precision** | 宏平均精确率 |
| **Recall** | 宏平均召回率 |
| **F1-Score** | 宏平均F1分数 |
| **ROC-AUC** | ROC曲线下面积 |
| **PR-AUC** | 精确率-召回率曲线下面积 |

### 可视化输出

- `acc_loss_over_epochs.png`：训练和验证损失/准确率曲线
- `roc_curve.png`：ROC曲线（多分类）
- `pr_curve.png`：精确率-召回率曲线

## 💾 数据管理

### 结果存储

系统支持多种结果存储方式：

1. **SQLite数据库**：`Inference/results/inference_results.db`
2. **MySQL数据库**：支持远程数据库连接
3. **CSV文件**：`Inference/results/inference_history.csv`

### 结果管理器功能

```python
from src.utils.result_manager import ResultManager

# 初始化（SQLite）
manager = ResultManager(db_type="sqlite")

# 初始化（MySQL）
manager = ResultManager(
    db_type="mysql",
    mysql_config={
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',
        'database': 'deeplearning_db'
    }
)

# 添加结果
manager.add_result_to_history(result_dict)

# 批量添加
manager.batch_add_results_to_history(results_list)

# 导出CSV
csv_path = manager.export_to_csv(results_list)

# 加载历史记录
history = manager.load_history(limit=100)
```

## 🔧 高级功能

### 持续扫描模式

自动监控Inference文件夹，检测新图片并自动处理：

```python
# 每2秒扫描一次
continuous_worker = ContinuousAutoInferenceWorker(
    inference_engine,
    inference_folder="Inference",
    ai_result_folder="AIResult",
    scan_interval=2
)
continuous_worker.start()
```

### 自动分类

根据推理结果自动将图片移动到对应文件夹：

```python
cls = result.get('class', 'NG')
dest_dir = Path(ai_result_folder) / cls
shutil.move(str(image_path), str(dest_dir / image_path.name))
```

## 📁 数据目录结构

### 训练数据

```
data/
└── raw/
    ├── OK/              # 合格品训练样本
    │   ├── 001.jpg
    │   └── ...
    └── NG/              # 不良品训练样本
        ├── 001.jpg
        └── ...
```

### 推理输入

```
Inference/
├── OK/                 # 合格品测试样本
├── NG/                 # 不良品测试样本
└── results/            # 推理结果存储
    ├── inference_history.csv
    ├── inference_results.db
    └── inference_results_YYYYMMDD_HHMMSS.csv
```

### 推理输出

```
AIResult/               # 自动分类输出
├── OK/                 # 判定为合格的图片
└── NG/                 # 判定为不良的图片
```

## 🎯 关键特性

### 1. 鲁棒性增强

- 解决工业路径中复杂字符导致的读取失败问题
- 支持多种图片格式（jpg, jpeg, png, bmp, tiff）
- 自动处理编码问题（UTF-8 with BOM）

### 2. 体验创新

- 打破算法"黑盒"，深度学习全生命周期日志可视化
- 实时显示训练损失、准确率曲线
- 推理结果表格化展示，支持颜色高亮

### 3. 闭环设计

- 数据采集 → 数据增强 → 模型训练 → 模型评估 → 推理分拣
- 完整的数据流和模型迭代流程

### 4. 工程化设计

- 多线程处理，UI不卡顿
- 支持GPU加速（CUDA）
- 模型权重自动保存和加载
- 支持断点续训

## 📝 代码示例

### 自定义训练

```python
from src.training.trainer import TrainModel

# 创建训练器
trainer = TrainModel(
    num_epochs=100,
    model_name='Swin_V2_B',
    batch_size=16,
    IMAGE_SIZE=(224, 224),
    Aug=0,
    patience=50,
    BEST_LOSS_THRESHOLD=0.0001
)

# 开始训练
trainer.train()
```

### 自定义推理

```python
from src.inference.inference_engine import InferenceEngine

# 初始化推理引擎
engine = InferenceEngine(
    model_name='Swin_V2_B',
    checkpoint='experiments/Swin_V2_B/checkpoints/best_Swin_V2_B_model.pth',
    IMAGE_SIZE=(224, 224)
)

# 单张推理
result = engine.infer('image.jpg')
print(f"预测类别: {result['class']}")
print(f"置信度: {result['confidence']:.2%}")

# 批量推理
results = engine.batch_infer(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### 模型测试

```python
from tests.testmodel import TestModel

tester = TestModel(
    model_name='Swin_V2_B',
    checkpoint='experiments/Swin_V2_B/checkpoints/best_Swin_V2_B_model.pth',
    IMAGE_SIZE=(224, 224),
    batch_size=16,
    Aug=0
)

tester.test_model()  # 输出测试指标和曲线图
```

## 🔮 未来扩展

### 深度方向

- [ ] 引入划痕、氧化、缺口等细分缺陷识别（多标签分类）
- [ ] 集成目标检测（YOLO/Faster R-CNN）定位缺陷位置
- [ ] 缺陷分割（U-Net/DeepLab）精确标记缺陷区域

### 性能方向

- [ ] 模型量化（INT8）和剪枝
- [ ] 部署至边缘计算设备（Jetson Nano/Raspberry Pi）
- [ ] 结合工业相机硬件接口（GigE/USB3 Vision）

### 智能方向

- [ ] 引入自动超参搜索（Optuna/Ray Tune）
- [ ] 主动学习策略，智能选择待标注样本
- [ ] 模型集成（Ensemble）提升鲁棒性

## 👥 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- torchvision提供的预训练模型
- 开源社区的各种工具和库

---

**作者**: 李毅  
**专业**: 智能科学与技术 B2212  
**学院**: 信息工程学院
