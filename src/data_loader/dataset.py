import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, batch_size, IMAGE_SIZE, Aug=1):
        # 添加 Aug 参数
        if not isinstance(Aug, int) or Aug not in (0, 1):
            raise ValueError("Aug must be 0 or 1")
        self.Aug = Aug
        
        if not isinstance(IMAGE_SIZE, (tuple, list)) or len(IMAGE_SIZE) != 2:
            raise ValueError("IMAGE_SIZE must be a tuple or list of two integers (height, width)")
        self.IMAGE_SIZE = IMAGE_SIZE
        self.batch_size = batch_size
        self.ROOT_FOLDER = "data/raw"
        # 根据 Aug 参数决定输出文件夹
        self.OUTPUT_FOLDER = "data/processed" if Aug else "data/raw"
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        self.data_mapping = self.get_image_paths_by_class(self.ROOT_FOLDER, extension="jpg")

        # 根据 Aug 参数定义不同的 transform
        if self.Aug:
            self.transform = A.Compose([
                A.Resize(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], p=1.0),
                A.OneOf([
                    A.Rotate(limit=0, p=0.25),
                    A.Rotate(limit=90, p=0.25),
                    A.Rotate(limit=180, p=0.25),
                    A.Rotate(limit=270, p=0.25),
                ], p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
                ToTensorV2()
            ])
        else: # 不使用数据增强，只进行必要的预处理
            self.transform = A.Compose([
                A.Resize(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
                ToTensorV2()
            ])
        
        if self.data_mapping:
            print("\n----------------------------------------")
            print(f"成功读取 {len(self.data_mapping)} 个类别:")
            
            all_image_paths = []
            all_labels = []
            
            for label, paths in self.data_mapping.items():
                print(f"  - 类别: {label}, 图像数量: {len(paths)}")
                for image_path in paths:
                    if self.Aug:
                        # 加载并增强图像 (仅在__init__中执行一次)
                        augmented_image = self.load_and_augment_image(image_path, self.transform)
                        
                        # 保存增强后的图像
                        output_subdir = os.path.join(self.OUTPUT_FOLDER, label)
                        os.makedirs(output_subdir, exist_ok=True)
                        output_image_path = os.path.join(output_subdir, os.path.basename(image_path))
                        self.save_image(augmented_image, output_image_path)
                        
                        # 将增强后的图像和标签添加到列表中
                        all_image_paths.append(output_image_path)  # 修改这里，使用增强后的图像路径
                    else:
                        # 如果不进行增强，直接使用原图像路径
                        all_image_paths.append(image_path)
                    all_labels.append(label)
                    
            print("图像增强完成！" if self.Aug else "图像路径读取完成！")
            print('------------------------------------')
            # 将所有标签转换为数字标签
            label_to_idx = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
            all_labels_idx = [label_to_idx[label] for label in all_labels]
            
            # 分割数据集为训练集、验证集和测试集
            train_image_paths, val_test_image_paths, train_labels, val_test_labels = train_test_split(
                all_image_paths, all_labels_idx, test_size=0.3, random_state=42
            )
            val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(
                val_test_image_paths, val_test_labels, test_size=1/3, random_state=42
            )
            
            print(f"训练集数量: {len(train_image_paths)}")
            print(f"验证集数量: {len(val_image_paths)}")
            print(f"测试集数量: {len(test_image_paths)}")

            self.train_image_paths = train_image_paths
            self.train_labels = train_labels
            self.val_image_paths = val_image_paths
            self.val_labels = val_labels
            self.test_image_paths = test_image_paths
            self.test_labels = test_labels

    def load_and_augment_image(self, image_path, transform):
        image = cv2.imread(image_path)  # 使用 OpenCV 读取图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将颜色空间从 BGR 转换为 RGB
        augmented_image = transform(image=image)["image"]  # 应用变换
        return augmented_image

    def save_image(self, augmented_image, output_path):
        augmented_image_np = augmented_image.numpy().transpose(1, 2, 0)  # 将 Tensor 转换为 NumPy 数组并调整通道顺序
        augmented_image_np = (augmented_image_np * 255).astype(np.uint8)  # 将像素值从 0-1 转换为 0-255
        plt.imsave(output_path, augmented_image_np)  # 保存图像

    def get_image_paths_by_class(self, root_dir, extension="jpg"):
        """
        遍历根目录下的子文件夹，将子文件夹名作为类别名，收集所有图片的路径。
        """
        if not os.path.isdir(root_dir):
            print(f"错误: 根目录不存在或不是一个目录: {root_dir}")
            return {}

        all_data = {}
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                category_label = class_name
                search_pattern = os.path.join(class_dir, f"*.{extension}")
                image_paths = glob(search_pattern)
                if not image_paths:
                    print(f"警告: 类别 '{category_label}' 文件夹下没有找到 *.{extension} 文件。")
                    continue
                all_data[category_label] = image_paths
                print(f"已读取类别 '{category_label}'，共 {len(image_paths)} 张图片。")

        return all_data

    # 将图像路径和标签转换为张量
    def paths_to_dataset(self, image_paths, labels):
        augmented_images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed_image = self.transform(image=image)["image"]
            augmented_images.append(transformed_image)
        return TensorDataset(torch.stack(augmented_images), torch.tensor(labels))

    def data_loader(self, batch_size):
        # 创建 DataLoader
        train_dataset = self.paths_to_dataset(self.train_image_paths, self.train_labels)
        val_dataset = self.paths_to_dataset(self.val_image_paths, self.val_labels)
        test_dataset = self.paths_to_dataset(self.test_image_paths, self.test_labels)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
