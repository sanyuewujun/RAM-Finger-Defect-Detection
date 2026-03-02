"""
推理应用快速启动脚本
支持命令行参数指定模型和推理文件夹
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='深度学习模型推理应用')
    parser.add_argument('--model', type=str, default='Vit_B_16', 
                       choices=['Vit_B_16', 'resnet50'],
                       help='选择模型 (默认: Vit_B_16)')
    parser.add_argument('--size', type=int, default=224,
                       help='图像大小 (默认: 224)')
    parser.add_argument('--folder', type=str, default='Inference',
                       help='推理文件夹路径 (默认: Inference)')
    
    args = parser.parse_args()
    
    # 验证推理文件夹
    inference_folder = Path(args.folder)
    inference_folder.mkdir(exist_ok=True)
    
    # 导入UI
    from ui.inference_ui import InferenceUI
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = InferenceUI()
    
    # 设置初始模型
    model_index = window.model_combo.findText(args.model)
    if model_index >= 0:
        window.model_combo.setCurrentIndex(model_index)
    
    # 设置图像大小
    window.size_spin.setValue(args.size)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
