# src/models/resnet50.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from src.data_loader.dataset import DataSet

class ModelSelection:
    def __init__(self,num_epochs, batch_size,IMAGE_SIZE,num_classes=2,):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.IMAGE_SIZE = IMAGE_SIZE
        self.batch_size = batch_size

    def select_model(self, model_name):
        if model_name == "resnet50":
            return self.resnet50_model()
        elif model_name == "Swin_V2_B":
            return self.Swin_V2_B_model()
        else:
            print(f"错误: 无效的模型名称 '{model_name}'。请选择 'ResNet50' 或 'Swin_V2_B'。")

    def resnet50_model(self):
        # 加载预训练的效果最好的resnet50权重版本
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 解冻 layer4 的参数
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 获取 ResNet50 最后一个全连接层 (model.fc) 的输入特征数
        num_ftrs = model.fc.in_features 

        # 修改分类层 (model.fc) 以适应你的分类任务
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, self.num_classes),  # 输出层：连接到你的类别数
        )

        # 打印模型结构
        print(model)

        # 打印可训练的参数数量 (用于确认冻结策略是否生效)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
        print(f"微调层参数量 (model.fc): {sum(p.numel() for p in model.fc.parameters()) / 1e6:.2f} M")

        return model

    def Swin_V2_B_model(self):
        # 使用效果最好的 Swin V2 B 预训练权重版本
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        # 冻结所有参数 (特征提取阶段)
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一个 Block (索引 11) 的参数
        for param in model.features[-1].parameters():
            param.requires_grad = True # 或者你的其他操作
        print("注意：已解冻最后一个 Transformer 编码器块的参数。")

        # 获取 Swin V2 B 分类头 (model.head) 的输入特征数
        num_ftrs = model.head.in_features

        # 修改分类头 (model.head) 以适应你的分类任务
        # Swin V2 B 的分类头就是模型在 `cls` token 上接的一个全连接层
        model.head = nn.Sequential(
            # 第一层从 Swin V2 B 的输出 (768) 开始
            nn.Linear(num_ftrs, 768),  # 注意：这里我们使用 2048 来匹配你 ResNet 示例中的结构
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # 输出层：连接到你的类别数
            nn.Linear(768, self.num_classes)
        )

        # 打印模型结构
        print(model)

        # 打印可训练的参数数量 (用于确认冻结策略是否生效)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
        print(f"微调层参数量 (model.head): {sum(p.numel() for p in model.head.parameters()) / 1e6:.2f} M")

        return model
