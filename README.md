文件结构介绍如下所示：

```
My_DL_Project/
├── data/                    # 原始数据和处理后的数据
│   ├── raw/                 # 原始、未修改的数据集
│   └── processed/           # 经过清洗、预处理、分割后的数据 (例如：训练集/验证集/测试集)
|
├── src/                     # 所有 Python 源代码 (模块化和可重用代码)
│   ├── __init__.py          # 使 src 目录成为一个 Python 包
│   ├── models/              # 模型定义文件
│   │   ├── __init__.py
│   │   └── resnet.py        # 具体的模型架构 (例如：ResNet-50)
│   ├── data_loader/         # 数据处理和加载模块
│   │   ├── __init__.py
│   │   └── dataset.py       # 自定义 Dataset 和 DataLoader
│   ├── training/            # 训练和评估相关的代码
│   │   ├── __init__.py
│   │   └── trainer.py       # 训练循环和评估函数
│   └── utils/               # 辅助函数和工具
│       ├── __init__.py
│       └── metrics.py       # 损失函数、评估指标等
|
├── experiments/             # 存储不同实验的配置和结果
│   ├── ...
│   └── resnet50      # 每次实验的独立文件夹
│       ├── checkpoints/     # 模型权重文件 (.pt, .h5)
│       ├── metrics/         # 测试过程中的指标记录
│       └── pltout/           # 测试结果的可视化输出
|
├── tests/                   # 单元测试和集成测试 (推荐使用 pytest)
│   ├── test_data.py
│   └── test_model.py
|
├── .gitignore               # 指定 Git 忽略的文件 (例如：data/raw, checkpoints/, venv/)
├── README.md                # 项目介绍、安装、运行和使用说明
├── requirements.txt         # 整个项目所需的所有依赖包
└── main.py                  # 程序的入口点，用于启动训练或评估
```