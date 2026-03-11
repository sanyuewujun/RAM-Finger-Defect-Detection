"""
结果分析工具 - 用于分析历史记录和计算指标
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.result_manager import ResultManager


class ResultAnalyzer:
    """结果分析类"""
    
    @staticmethod
    def extract_true_label_from_filename(filename: str) -> str:
        """
        从文件名中提取真实标签
        支持的格式: OK001.jpg, NG_001.jpg 等
        
        Args:
            filename: 文件名
            
        Returns:
            str: 提取的标签（如 'OK' 或 'NG'）
        """
        name = Path(filename).stem  # 去掉扩展名
        
        # 尝试从开头提取
        for label in ['OK', 'NG', 'GOOD', 'BAD', 'PASS', 'FAIL']:
            if name.upper().startswith(label):
                return label
        
        # 尝试从中间提取 (如 img_OK_001)
        for label in ['OK', 'NG', 'GOOD', 'BAD', 'PASS', 'FAIL']:
            if '_' + label in name.upper():
                return label
        
        return 'N/A'
    
    @staticmethod
    def analyze_history_file(history_file: Path, output_file: Path = None) -> Dict:
        """
        分析历史记录文件并计算精度指标
        
        Args:
            history_file: 历史记录文件路径
            output_file: 输出文件路径（可选）
            
        Returns:
            dict: 分析结果
        """
        results = []
        true_labels = []
        
        try:
            with open(history_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = row.get('图片名称', '')
                    pred_class = row.get('预测类别', 'N/A')
                    
                    # 从文件名提取真实标签
                    true_label = ResultAnalyzer.extract_true_label_from_filename(image_name)
                    
                    results.append({
                        'image_path': image_name,
                        'class': pred_class,
                        'confidence': float(row.get('置信度', 0)) if row.get('置信度') else 0
                    })
                    true_labels.append(true_label)
            
            # 计算指标
            result_manager = ResultManager()
            metrics = result_manager.calculate_metrics(results, true_labels)
            
            # 输出信息
            analysis_text = f"\n历史记录分析结果\n{'='*60}\n"
            analysis_text += f"文件: {history_file}\n"
            analysis_text += f"记录数: {metrics['total']}\n"
            analysis_text += f"成功预测: {metrics['correct']}\n"
            analysis_text += f"总体准确率 (Accuracy): {metrics['accuracy']:.4f}\n"
            analysis_text += f"加权平均精准率 (Precision): {metrics.get('weighted_precision', 0):.4f}\n\n"
            
            analysis_text += f"各类别详细指标:\n"
            for cls in sorted(metrics['class_wise_metrics'].keys()):
                cls_metrics = metrics['class_wise_metrics'][cls]
                analysis_text += f"  {cls}:\n"
                analysis_text += f"    真阳性 (TP): {cls_metrics['tp']}\n"
                analysis_text += f"    假阳性 (FP): {cls_metrics['fp']}\n"
                analysis_text += f"    假阴性 (FN): {cls_metrics['fn']}\n"
                analysis_text += f"    精准率 (Precision): {cls_metrics['precision']:.4f}\n"
                analysis_text += f"    召回率 (Recall): {cls_metrics['recall']:.4f}\n"
                analysis_text += f"    F1分数: {cls_metrics['f1_score']:.4f}\n"
            
            # 写入输出文件
            if output_file:
                with open(output_file, 'w', encoding='utf-8-sig') as f:
                    f.write(analysis_text)
                print(f"分析结果已保存到: {output_file}")
            
            print(analysis_text)
            return metrics
            
        except Exception as e:
            print(f"分析失败: {e}")
            return {}
    
    @staticmethod
    def update_history_with_true_labels(history_file: Path) -> bool:
        """
        更新历史记录文件，根据文件名填充真实标签列
        
        Args:
            history_file: 历史记录文件路径
            
        Returns:
            bool: 是否成功更新
        """
        try:
            rows = []
            with open(history_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = row.get('图片名称', '')
                    # 如果真实标签为空，从文件名提取
                    if not row.get('真实标签') or row.get('真实标签') == 'N/A':
                        true_label = ResultAnalyzer.extract_true_label_from_filename(image_name)
                        row['真实标签'] = true_label
                        
                        # 更新是否正确
                        pred_class = row.get('预测类别', 'N/A')
                        if pred_class != 'N/A' and true_label != 'N/A':
                            row['是否正确'] = '是' if pred_class == true_label else '否'
                    rows.append(row)
            
            # 写回文件
            if rows:
                headers = list(rows[0].keys())
                with open(history_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"已更新 {len(rows)} 条记录")
                return True
            return False
            
        except Exception as e:
            print(f"更新失败: {e}")
            return False


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(description='推理结果分析工具')
    parser.add_argument('--history', type=str, default='Inference/results/inference_history.csv',
                       help='历史记录文件路径')
    parser.add_argument('--analyze', action='store_true',
                       help='分析历史记录文件并计算精度指标')
    parser.add_argument('--update-labels', action='store_true',
                       help='自动从文件名提取真实标签并更新历史记录')
    parser.add_argument('--output', type=str, help='分析结果输出文件路径')
    
    args = parser.parse_args()
    history_file = Path(args.history)
    
    if not history_file.exists():
        print(f"错误: 找不到历史记录文件 {history_file}")
        return
    
    if args.analyze:
        output_file = Path(args.output) if args.output else None
        ResultAnalyzer.analyze_history_file(history_file, output_file)
    elif args.update_labels:
        ResultAnalyzer.update_history_with_true_labels(history_file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
