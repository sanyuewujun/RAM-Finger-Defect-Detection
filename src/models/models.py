# src/models/resnet50.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

class ModelSelection:
    def __init__(self, num_epochs, batch_size, IMAGE_SIZE, num_classes=2):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.IMAGE_SIZE = IMAGE_SIZE
        self.batch_size = batch_size

    def select_model(self, model_name):
        if model_name == "resneXt101(32x8d)":
            return self.resnext101_32x8d_model() 
        elif model_name == "ResNeXt50_32X4D":
            return self.resnext50_32x4d_model()
        elif model_name == "ResNeXt101_64X4D":
            return self.resnext101_64x4d_model()
        elif model_name == "Swin_B":
            return self.Swin_B_model()
        elif model_name == "Swin_V2_B":
            return self.Swin_V2_B_model()
        else:
            print(f"错误: 无效的模型名称 '{model_name}'。请选择 'resneXt101(32x8d)' 或 'Swin_V2_B'。")

    def resnext50_32x4d_model(self):
        # 1. 加载预训练权重
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 2. 冻结所有参数 (Backbone冻结)
        for param in model.parameters():
            param.requires_grad = False

        # 3. 解冻 layer4 的参数 (深层特征微调)
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 4. 获取最后一个全连接层的输入特征数
        num_ftrs = model.fc.in_features 

        # 5. 修改分类层 (model.fc) 
        # 注意：PyTorch中新实例化的层(nn.Linear等)默认 requires_grad=True，无需额外解冻
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, self.num_classes),  # 输出层：连接到你的类别数
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in model.fc.parameters())

        print(f"├─ 模型总参数量: {total_params / 1e6:.2f} M")
        print(f"├─ 可训练参数量: {trainable_params / 1e6:.2f} M (包含 layer4 和 新的 fc)")
        print(f"└─ 自定义分类头参数量: {fc_params / 1e6:.2f} M\n")

        return model
    
    def resnext101_32x8d_model(self):
        # 1. 加载预训练权重 (使用正确的函数名 resnext101_32x8d)
        model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)

        # 2. 冻结所有参数 (Backbone冻结)
        for param in model.parameters():
            param.requires_grad = False

        # 3. 解冻 layer4 的参数 (深层特征微调)
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 4. 获取最后一个全连接层的输入特征数
        num_ftrs = model.fc.in_features 

        # 5. 修改分类层 (model.fc) 
        # 注意：PyTorch中新实例化的层(nn.Linear等)默认 requires_grad=True，无需额外解冻
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, self.num_classes),  # 输出层：连接到你的类别数
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in model.fc.parameters())

        print(f"├─ 模型总参数量: {total_params / 1e6:.2f} M")
        print(f"├─ 可训练参数量: {trainable_params / 1e6:.2f} M (包含 layer4 和 新的 fc)")
        print(f"└─ 自定义分类头参数量: {fc_params / 1e6:.2f} M\n")

        return model
    
    def resnext101_64x4d_model(self):
        # 1. 加载预训练权重 (使用正确的函数名 resnext101_64x4d)
        model = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT)

        # 2. 冻结所有参数 (Backbone冻结)
        for param in model.parameters():
            param.requires_grad = False

        # 3. 解冻 layer4 的参数 (深层特征微调)
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 4. 获取最后一个全连接层的输入特征数
        num_ftrs = model.fc.in_features 

        # 5. 修改分类层 (model.fc) 
        # 注意：PyTorch中新实例化的层(nn.Linear等)默认 requires_grad=True，无需额外解冻
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, self.num_classes),  # 输出层：连接到你的类别数
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fc_params = sum(p.numel() for p in model.fc.parameters())

        print(f"├─ 模型总参数量: {total_params / 1e6:.2f} M")
        print(f"├─ 可训练参数量: {trainable_params / 1e6:.2f} M (包含 layer4 和 新的 fc)")
        print(f"└─ 自定义分类头参数量: {fc_params / 1e6:.2f} M\n")

        return model
    
    def Swin_B_model(self):
        # 使用效果最好的 Swin B 预训练权重版本
        model = swin_b(weights=Swin_B_Weights.DEFAULT)
        # 冻结所有参数 (特征提取阶段)
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一个 Block (索引 3) 的参数
        for param in model.features[-1].parameters():
            param.requires_grad = True # 或者你的其他操作
        print("注意：已解冻最后一个 Transformer 编码器块的参数。")

        # 获取 Swin B 分类头 (model.head) 的输入特征数
        num_ftrs = model.head.in_features

        # 修改分类头 (model.head) 以适应你的分类任务
        # Swin B 的分类头就是模型在 `cls` token 上接的一个全连接层
        model.head = nn.Sequential(
            # 第一层从 Swin B 的输出 (768) 开始
            nn.Linear(num_ftrs, 768),  # 注意：这里我们使用 2048 来匹配你 ResNet 示例中的结构
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # 输出层：连接到你的类别数
            nn.Linear(768, self.num_classes)
        )

        # 打印可训练的参数数量 (用于确认冻结策略是否生效)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
        print(f"微调层参数量 (model.head): {sum(p.numel() for p in model.head.parameters()) / 1e6:.2f} M")

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

        # 打印可训练的参数数量 (用于确认冻结策略是否生效)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型总参数量: {total_params / 1e6:.2f} M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
        print(f"微调层参数量 (model.head): {sum(p.numel() for p in model.head.parameters()) / 1e6:.2f} M")

        return model
