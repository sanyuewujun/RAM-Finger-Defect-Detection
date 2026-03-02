"""
推理引擎 - 用于加载模型并进行单张图片推理
"""
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.models import ModelSelection


class InferenceEngine:
    """推理引擎类"""
    
    def __init__(self, model_name, checkpoint, IMAGE_SIZE=(224, 224)):
        """
        初始化推理引擎
        
        Args:
            model_name: 模型名称 ('resnet50' 或 'Vit_B_16')
            checkpoint: 模型权重文件路径
            IMAGE_SIZE: 输入图像大小
        """
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.IMAGE_SIZE = IMAGE_SIZE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = None
        
        # 初始化模型和变换
        self._init_model()
        self._init_transform()
    
    def _init_model(self):
        """初始化模型"""
        try:
            # 获取类别数量
            data_folder = 'data/raw'
            if os.path.exists(data_folder):
                num_classes = len(os.listdir(data_folder))
                self.class_names = sorted(os.listdir(data_folder))
            else:
                num_classes = 2
                self.class_names = ['NG', 'OK']
            
            # 创建模型
            model_selector = ModelSelection(
                num_epochs=0, 
                batch_size=1, 
                IMAGE_SIZE=self.IMAGE_SIZE, 
                num_classes=num_classes
            )
            self.model = model_selector.select_model(self.model_name)
            
            # 加载权重
            if os.path.exists(self.checkpoint):
                checkpoint_data = torch.load(self.checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint_data)
                print(f"✓ 成功加载模型权重: {self.checkpoint}")
            else:
                raise FileNotFoundError(f"找不到模型文件: {self.checkpoint}")
            
            # 设置模型为评估模式
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")
    
    def _init_transform(self):
        """初始化图像变换"""
        self.transform = A.Compose([
            A.Resize(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ])
    
    def infer(self, image_path):
        """
        对单张图片进行推理
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含预测结果的字典
                {
                    'class': 预测类别,
                    'confidence': 置信度,
                    'probabilities': 各类别概率,
                    'class_names': 类别名称列表
                }
        """
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 获取结果
            predicted_class = predicted.item()
            predicted_name = self.class_names[predicted_class]
            confidence_score = confidence.item()
            probs_dict = {
                name: prob.item() 
                for name, prob in zip(self.class_names, probabilities[0])
            }
            
            return {
                'class': predicted_name,
                'class_id': predicted_class,
                'confidence': confidence_score,
                'probabilities': probs_dict,
                'class_names': self.class_names
            }
            
        except Exception as e:
            raise RuntimeError(f"推理失败: {str(e)}")
    
    def batch_infer(self, image_paths):
        """
        对多张图片进行批量推理
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            list: 推理结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.infer(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        return results
