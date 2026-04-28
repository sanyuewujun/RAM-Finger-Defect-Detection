import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.models.models import ModelSelection
from src.data_loader.dataset import DataSet
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import csv  # 添加csv库以处理CSV文件
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self, num_epochs, model_name, batch_size, IMAGE_SIZE, Aug, patience=50, BEST_LOSS_THRESHOLD=0.0001, num_classes=2):
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.batch_size = batch_size
        self.IMAGE_SIZE = IMAGE_SIZE
        self.patience = patience
        self.BEST_LOSS_THRESHOLD = BEST_LOSS_THRESHOLD
        self.best_val_loss = float('inf')  # 初始最佳验证损失设置为无穷大
        self.epochs_no_improve = 0  # 记录没有改善的epoch数量
        self.num_classes = num_classes
        self.Aug = Aug
        # 下列变量用于记录最佳模型参数
        self.best_epoch = 0
        self.best_train_acc = 0
        self.best_train_loss = 0
        self.best_val_acc = 0
        self.best_lr = 0

    def train(self):
        dataset = DataSet(self.batch_size, self.IMAGE_SIZE, self.Aug)
        train_loader, val_loader, _ = dataset.data_loader(self.batch_size)
        modelselection = ModelSelection(self.num_epochs, self.batch_size, self.IMAGE_SIZE)
        model = modelselection.select_model(self.model_name)
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

        # 定义设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义损失函数,使用CrossEntropyloss时可以不在nn末层中添加激活函数
        criterion = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()
        if self.model_name == "resneXt101(32x8d)":
            # 自定义优化器和学习率调度器
            LR_CUSTOM_HEAD = 1e-4  # 自定义隐藏层
            LR_LAYER4 = 1e-5  # 解冻的 layer4
            # 定义优化器参数组
            param_groups = [
                {'params': model.fc.parameters(), 'lr': LR_CUSTOM_HEAD, 'group_name': 'custom_head'},
                {'params': model.layer4.parameters(), 'lr': LR_LAYER4, 'group_name': 'layer4_fine_tune'}
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            steps_per_epoch = len(train_loader)
            # 定义学习率调度器
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[LR_CUSTOM_HEAD, LR_LAYER4],
                steps_per_epoch=steps_per_epoch,
                epochs=self.num_epochs,
                pct_start=0.1
            )
            self.best_lr = LR_CUSTOM_HEAD  # 记录学习率
            print(f"Successful!{self.model_name}学习率调度策略已设置。")
            
        elif self.model_name == "ResNeXt50_32X4D":
            # 自定义优化器和学习率调度器
            LR_CUSTOM_HEAD = 1e-4  # 自定义隐藏层
            LR_LAYER4 = 1e-5  # 解冻的 layer4
            # 定义优化器参数组
            param_groups = [
                {'params': model.fc.parameters(), 'lr': LR_CUSTOM_HEAD, 'group_name': 'custom_head'},
                {'params': model.layer4.parameters(), 'lr': LR_LAYER4, 'group_name': 'layer4_fine_tune'}
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            steps_per_epoch = len(train_loader)
            # 定义学习率调度器
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[LR_CUSTOM_HEAD, LR_LAYER4],
                steps_per_epoch=steps_per_epoch,
                epochs=self.num_epochs,
                pct_start=0.1
            )
            self.best_lr = LR_CUSTOM_HEAD  # 记录学习率
            print(f"Successful!{self.model_name}学习率调度策略已设置。")
            
        elif self.model_name == "ResNeXt101_64X4D":
            # 自定义优化器和学习率调度器
            LR_CUSTOM_HEAD = 1e-4  # 自定义隐藏层
            LR_LAYER4 = 1e-5  # 解冻的 layer4
            # 定义优化器参数组
            param_groups = [
                {'params': model.fc.parameters(), 'lr': LR_CUSTOM_HEAD, 'group_name': 'custom_head'},
                {'params': model.layer4.parameters(), 'lr': LR_LAYER4, 'group_name': 'layer4_fine_tune'}
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            steps_per_epoch = len(train_loader)
            # 定义学习率调度器
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[LR_CUSTOM_HEAD, LR_LAYER4],
                steps_per_epoch=steps_per_epoch,
                epochs=self.num_epochs,
                pct_start=0.1
            )
            self.best_lr = LR_CUSTOM_HEAD  # 记录学习率
            print(f"Successful!{self.model_name}学习率调度策略已设置。")
            
        elif self.model_name == "Swin_B":
            # 自定义优化器和学习率调度器
            LR_CUSTOM_HEAD = 1e-4  # 自定义分类头，可以稍高
            LR_LAST_BLOCK = 1e-5  # 解冻的最后一个 Transformer Block，非常低
            # 定义优化器参数组
            param_groups = [
                {'params': model.head.parameters(), 'lr': LR_CUSTOM_HEAD, 'group_name': 'custom_head'},
                {'params': model.features[-1].parameters(), 'lr': LR_LAST_BLOCK, 'group_name': 'last_block_fine_tune'}
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            steps_per_epoch = len(train_loader)
            # 定义学习率调度器
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[LR_CUSTOM_HEAD, LR_LAST_BLOCK],
                steps_per_epoch=steps_per_epoch,
                epochs=self.num_epochs,
                pct_start=0.1
            )
            self.best_lr = LR_CUSTOM_HEAD  # 记录学习率
            print(f"Successful!{self.model_name}学习率调度策略已设置。")

        elif self.model_name == "Swin_V2_B":
            # 自定义优化器和学习率调度器
            LR_CUSTOM_HEAD = 1e-4  # 自定义分类头，可以稍高
            LR_LAST_BLOCK = 1e-5  # 解冻的最后一个 Transformer Block，非常低
            # 定义优化器参数组
            param_groups = [
                {'params': model.head.parameters(), 'lr': LR_CUSTOM_HEAD, 'group_name': 'custom_head'},
                {'params': model.features[-1].parameters(), 'lr': LR_LAST_BLOCK, 'group_name': 'last_block_fine_tune'}
            ]

            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            steps_per_epoch = len(train_loader)
            # 定义学习率调度器
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[LR_CUSTOM_HEAD, LR_LAST_BLOCK],
                steps_per_epoch=steps_per_epoch,
                epochs=self.num_epochs,
                pct_start=0.1
            )
            self.best_lr = LR_CUSTOM_HEAD  # 记录学习率
            print(f"Successful!{self.model_name}学习率调度策略已设置。")
        else:
            pass

        for epoch in range(self.num_epochs):
            # 训练阶段
            model.to(device)
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(train_accuracy)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            val_epoch_loss = val_loss / len(val_loader)
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_accuracy)

            print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            # 使用策略 1 检查是否达到预设的最低损失阈值
            if self.strategy_one(val_epoch_loss, model):
                break

            # 使用策略 2 标准的基于 patience 的早停
            if self.strategy_two(val_epoch_loss, model, epoch, train_accuracy, epoch_loss, val_accuracy):
                continue
            else:
                self.epochs_no_improve += 1
                print(f"验证损失未改善，耐心计数: {self.epochs_no_improve}/{self.patience}")

            if self.epochs_no_improve >= self.patience:
                print(f"\n连续 {self.patience} 个epoch验证损失没有改善，停止训练。")
                break

        # 绘制训练损失和验证损失
        self.plot_results(train_losses, val_losses, train_accuracies, val_accuracies)

        # 保存最佳模型的信息
        self.save_best_model_info()

    def strategy_one(self, val_epoch_loss, model):
        if val_epoch_loss <= self.BEST_LOSS_THRESHOLD:
            print(f"\n✨ 验证损失 {val_epoch_loss:.4f} 达到或低于阈值 {self.BEST_LOSS_THRESHOLD}，停止训练。")
            torch.save(model.state_dict(), f"experiments/{self.model_name}/checkpoints/best_{self.model_name}_model_threshold_reached.pth")
            print("模型已保存。")
            return True
        return False

    def strategy_two(self, val_epoch_loss, model, epoch, train_accuracy, epoch_loss, val_accuracy):
        if val_epoch_loss < self.best_val_loss:
            self.best_val_loss = val_epoch_loss
            self.best_epoch = epoch + 1  # 记录最佳模型的epoch
            self.best_train_acc = train_accuracy  # 记录最佳模型的训练准确率
            self.best_train_loss = epoch_loss  # 记录最佳模型的训练损失
            self.best_val_acc = val_accuracy  # 记录最佳模型的验证准确率
            self.epochs_no_improve = 0
            # 保存最佳模型
            # 2. 检查并创建目录（核心修复）
            save_dir = f"experiments/{self.model_name}/checkpoints"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), f"experiments/{self.model_name}/checkpoints/best_{self.model_name}_model.pth")
            print("验证损失降低，保存最佳模型。")
            return True
        return False

    def plot_results(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # 绘制训练准确率和验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        # 显示图形
        plt.tight_layout()

        # 创建保存文件夹并保存图片
        save_dir = os.path.join(f'experiments/{self.model_name}', 'pltout')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'acc_loss_over_epochs.png')
        plt.savefig(save_path)
        plt.show()

    def save_best_model_info(self):
        # 记录相关信息到CSV文件
        file_path = os.path.join(f'experiments/{self.model_name}', 'best_model_info.csv')
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Validation Accuracy', 'Validation Loss', 'Learning Rate', 'Batch Size', 'Augmentation'])
            writer.writerow([self.best_epoch, self.best_train_acc, self.best_train_loss, self.best_val_acc, self.best_val_loss, self.best_lr, self.batch_size, self.Aug])
        print(f"最佳模型的信息已保存到 {file_path}")
