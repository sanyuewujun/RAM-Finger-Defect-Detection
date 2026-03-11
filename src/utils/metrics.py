import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import csv  # 导入csv模块

class Metrics:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names
        self.label_map = {i: name for i, name in enumerate(self.class_names)} if self.class_names else {}

    def metrics(self, all_preds, all_labels, all_outputs, num_classes, running_loss, total, correct):
        # accuracy computation (loss still computed if caller provides it, but not reported)
        test_acc = 100 * correct / total

        # 计算多分类的precision、recall和f1-score，并转换为百分比
        precision = precision_score(all_labels, all_preds, average='macro') * 100
        recall = recall_score(all_labels, all_preds, average='macro') * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100

        # 绘制ROC曲线和PR曲线
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.eye(num_classes)[np.array(all_labels)]
        self.plot_roc_curve(all_outputs, all_labels, num_classes)
        self.plot_pr_curve(all_outputs, all_labels, num_classes)

        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Precision: {precision:.2f}%')
        print(f'Recall: {recall:.2f}%')
        print(f'F1 Score: {f1:.2f}%')

        # 将指标保存到CSV文件
        save_dir = os.path.join(f'experiments/{self.model}', 'test_metrics')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'test_results.csv')

        # 写入CSV文件
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'])  # 写入表头
            writer.writerow([self.model, f'{test_acc:.2f}%', f'{precision:.2f}%', f'{recall:.2f}%', f'{f1:.2f}%'])
        
        # 在所有文件操作完成后，显示所有绘图窗口
        plt.show()

    def plot_roc_curve(self, outputs, labels, num_classes):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], outputs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 画图
        plt.figure()
        for i in range(num_classes):
            class_label = self.label_map.get(i, f'Class {i}')
            plt.plot(fpr[i], tpr[i], label=f'{class_label} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # 创建保存文件夹并保存图片
        save_dir = os.path.join(f'experiments/{self.model}', 'pltout')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_curve.png')
        plt.savefig(save_path)

    def plot_pr_curve(self, outputs, labels, num_classes):
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(labels[:, i], outputs[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # 画图
        plt.figure()
        for i in range(num_classes):
            class_label = self.label_map.get(i, f'Class {i}')
            plt.plot(recall[i], precision[i], label=f'{class_label} (area = {pr_auc[i]:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")

        # 创建保存文件夹并保存图片
        save_dir = os.path.join(f'experiments/{self.model}', 'pltout')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'pr_curve.png')
        plt.savefig(save_path)
