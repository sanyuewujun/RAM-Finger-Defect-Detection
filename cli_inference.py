"""
命令行推理脚本 - 无需UI的快速推理
用法:
    python cli_inference.py image.jpg --model Vit_B_16
    python cli_inference.py folder/ --model resnet50  # 批量推理
"""
import sys
import argparse
from pathlib import Path
from src.inference.inference_engine import InferenceEngine
import json
from datetime import datetime

def infer_single_image(engine, image_path):
    """推理单张图片"""
    try:
        result = engine.infer(image_path)
        result['image_path'] = str(image_path)
        return result
    except Exception as e:
        return {'image_path': str(image_path), 'error': str(e)}

def print_result(result):
    """打印推理结果"""
    if 'error' in result:
        print(f"❌ {result['image_path']}: {result['error']}")
        return
    
    image_name = Path(result['image_path']).name
    pred_class = result['class']
    confidence = result['confidence']
    
    print(f"\n{'='*60}")
    print(f"📷 图片: {image_name}")
    print(f"{'='*60}")
    print(f"🎯 预测类别: {pred_class}")
    print(f"📊 置信度:   {confidence:.2%}")
    print(f"\n各类别概率:")
    for class_name, prob in result['probabilities'].items():
        bar_length = int(prob * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {class_name:10s} {bar} {prob:6.2%}")

def main():
    parser = argparse.ArgumentParser(
        description='命令行模型推理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  推理单张图片:
    python cli_inference.py image.jpg --model Vit_B_16
    
  批量推理文件夹:
    python cli_inference.py ./images/ --model resnet50
    
  导出结果为JSON:
    python cli_inference.py image.jpg --output result.json
        """
    )
    
    parser.add_argument('input', type=str,
                       help='输入图片路径或文件夹路径')
    parser.add_argument('--model', type=str, default='Vit_B_16',
                       choices=['Vit_B_16', 'resnet50'],
                       help='选择模型 (默认: Vit_B_16)')
    parser.add_argument('--size', type=int, default=224,
                       help='图像大小 (默认: 224)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径 (JSON格式)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点文件路径')
    
    args = parser.parse_args()
    
    # 确定检查点文件
    checkpoint = args.checkpoint or f'experiments/{args.model}/checkpoints/best_{args.model}_model.pth'
    
    # 初始化引擎
    print(f"🔧 初始化模型: {args.model}")
    try:
        engine = InferenceEngine(args.model, checkpoint, (args.size, args.size))
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 处理输入
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        # 单张图片
        print(f"🔍 推理中: {input_path.name}")
        result = infer_single_image(engine, input_path)
        print_result(result)
        results.append(result)
        
    elif input_path.is_dir():
        # 文件夹批量推理
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = [f for f in input_path.iterdir() if f.suffix in image_extensions]
        
        if not image_files:
            print(f"❌ 文件夹中找不到支持的图片文件")
            sys.exit(1)
        
        print(f"🔍 找到 {len(image_files)} 张图片，开始推理...\n")
        
        for idx, image_file in enumerate(sorted(image_files), 1):
            result = infer_single_image(engine, image_file)
            print_result(result)
            results.append(result)
            print(f"进度: {idx}/{len(image_files)}")
    else:
        print(f"❌ 输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"📈 推理统计")
    print(f"{'='*60}")
    print(f"总图片数: {len(results)}")
    
    success_count = sum(1 for r in results if 'error' not in r)
    print(f"成功推理: {success_count}")
    print(f"失败推理: {len(results) - success_count}")
    
    if success_count > 0:
        class_counts = {}
        total_confidence = 0
        for result in results:
            if 'error' not in result:
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_confidence += result['confidence']
        
        print(f"平均置信度: {total_confidence / success_count:.2%}")
        print(f"\n类别分布:")
        for class_name, count in sorted(class_counts.items()):
            percentage = count / success_count * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # 导出结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为JSON可序列化的格式
        json_results = []
        for result in results:
            json_result = {
                'image_path': result.get('image_path'),
                'class': result.get('class'),
                'confidence': result.get('confidence'),
                'probabilities': result.get('probabilities'),
                'error': result.get('error'),
                'timestamp': datetime.now().isoformat()
            }
            json_results.append(json_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已导出: {output_path}")

if __name__ == '__main__':
    main()
