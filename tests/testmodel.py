import torch
import torch.nn as nn
import os
from src.models.models import ModelSelection
from src.data_loader.dataset import DataSet
from src.utils.metrics import Metrics
class TestModel:
    def __init__(self, model_name, checkpoint, IMAGE_SIZE, batch_size, Aug):
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.IMAGE_SIZE = IMAGE_SIZE
        self.batch_size = batch_size
        self.Aug = Aug

    def test_model(self):
        # 使用 ModelSelection 创建模型以确保与训练时结构一致
        num_classes = len(os.listdir('data/raw'))
        model_selector = ModelSelection(num_epochs=0, batch_size=self.batch_size, IMAGE_SIZE=self.IMAGE_SIZE, num_classes=num_classes)
        model = model_selector.select_model(self.model_name)

        # 加载检查点文件中的权重
        model.load_state_dict(torch.load(self.checkpoint, weights_only=True))
        model.eval()

        # 创建数据集加载器
        dataset = DataSet(self.batch_size, self.IMAGE_SIZE, self.Aug)
        _, _, test_loader = dataset.data_loader(self.batch_size)

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_outputs = []
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0

        if torch.cuda.is_available():
            model.cuda()
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                all_outputs.append(outputs.cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        
        # 计算并打印指标
        # 获取按字母顺序排序的类名，以匹配DataSet中的标签编码
        class_names = sorted(os.listdir('data/raw'))
        
        # 创建Metrics实例并传入类名
        metrics_calculator = Metrics(model=self.model_name, class_names=class_names)
        metrics_calculator.metrics(all_preds, all_labels, all_outputs, num_classes, running_loss, total, correct)
