import os
from src.training.trainer import TrainModel
from tests.testmodel import TestModel

# --- 运行模式 ---
# 可选: 'train', 'test'
MODE = 'train'

# --- 模型配置 ---
# 可选模型: 'resnet50','Vit_B_16'
model_name = 'resnet50'

# --- 数据集配置 ---
IMAGE_SIZE = (224, 224)  # 图像尺寸(224,224): 'resnet50','Vit_B_16'
Aug = 0         # 是否启用数据增强 (1: 是, 0: 否)

# --- 训练配置 ---
# 仅在 MODE = 'train' 时生效
num_epochs = 1         # 训练轮次
batch_size = 16          # 批处理大小 (可根据显存调整)

# --- 测试配置 ---
# 仅在 MODE = 'test' 时生效
# 指定要加载的模型权重文件路径
checkpoint = f'experiments/{model_name}/checkpoints/best_{model_name}_model.pth'


# =================================================================
# 主函数
# =================================================================
def main():
    """
    根据配置参数执行训练或测试任务。
    """
    if MODE == 'train':
        print(f"--- 启动训练模式: {model_name} ---")
        trainer = TrainModel(num_epochs,model_name,batch_size,IMAGE_SIZE,Aug)
        trainer.train()
    elif MODE == 'test':
        print(f"--- 启动测试模式: {model_name} ---")
        tester = TestModel(model_name, checkpoint, IMAGE_SIZE, batch_size, Aug)
        tester.test_model()
    else:
        print(f"错误: 无效的模式 '{MODE}'。请选择 'train' 或 'test'。")

if __name__ == '__main__':
    main()
