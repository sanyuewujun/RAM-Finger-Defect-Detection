"""
推理结果管理模块 - 处理结果保存、导出和统计
"""
import csv
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# MySQL支持（可选）
try:
    import pymysql
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

# pandas是可选的，用于加载历史记录
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ResultManager:
    """推理结果管理类"""
    
    # CSV表头定义
    CSV_HEADERS = [
        '推理时间',          # 推理执行的时间戳
        '图片名称',          # 图片文件名
        'AI判定结果',        # 模型预测的类别
        '置信率',            # 预测置信度
        '推理耗时(秒)',      # 单张图片推理耗时
        '图片路径',          # 完整图片路径
    ]
    OLD_CSV_HEADERS = [
        '推理时间',
        '图片名称',
        '预测类别',
        '置信度',
        '推理耗时(秒)',
        '图片路径',
        '真实标签'
    ]
    
    def __init__(self, output_folder: str = "Inference/results", db_type: str = "sqlite", 
                 mysql_config: Dict = None):
        """
        初始化结果管理器
        
        Args:
            output_folder: 输出文件夹路径
            db_type: 数据库类型 ('sqlite' 或 'mysql')
            mysql_config: MySQL配置字典，包含：
                - host: 主机地址
                - port: 端口号
                - user: 用户名
                - password: 密码
                - database: 数据库名
                - charset: 字符集（默认'utf8mb4'）
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_folder / "inference_history.csv"
        self.db_type = db_type.lower()
        
        if self.db_type == "mysql":
            if not HAS_MYSQL:
                raise ImportError("MySQL支持需要安装 pymysql: pip install pymysql")
            self.mysql_config = mysql_config or {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '',
                'database': 'deeplearning_db',
                'charset': 'utf8mb4'
            }
            self.db_file = None  # MySQL不需要文件路径
        else:
            self.db_file = self.output_folder / "inference_results.db"
            self.mysql_config = None
        
        # 初始化历史文件和数据库
        self._init_history_file()
        self._init_database()
    
    def _init_history_file(self):
        """初始化历史记录文件"""
        try:
            if not self.history_file.exists():
                with open(self.history_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADERS)
            else:
                with open(self.history_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if rows and rows[0] == self.OLD_CSV_HEADERS:
                    new_rows = [self.CSV_HEADERS]
                    for row in rows[1:]:
                        if not row:
                            continue
                        row = row + [''] * (len(self.OLD_CSV_HEADERS) - len(row))
                        new_rows.append([
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[7]
                        ])
                    with open(self.history_file, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        writer.writerows(new_rows)
        except Exception as e:
            print(f"初始化历史文件失败: {e}")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            if self.db_type == "mysql":
                self._init_mysql_database()
            else:
                self._init_sqlite_database()
        except Exception as e:
            print(f"初始化数据库失败: {e}")
    
    def _init_sqlite_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        
        # 检查表是否已存在以及结构是否正确
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inference_results'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # 检查表结构
            cursor.execute("PRAGMA table_info(inference_results)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # 如果表结构不匹配，删除并重新创建
            expected_columns = ['id', '推理时间', '图片名称', 'AI判定结果', 
                              '置信率', '推理耗时', '图片路径', '是否正确']
            if set(column_names) != set(expected_columns):
                print("数据库表结构不匹配，正在重新创建...")
                cursor.execute("DROP TABLE inference_results")
                table_exists = None
        
        if not table_exists:
            # 创建推理结果表
            cursor.execute('''
                CREATE TABLE inference_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    推理时间 DATETIME NOT NULL,
                    图片名称 VARCHAR(256) NOT NULL,
                    AI判定结果 VARCHAR(256) NOT NULL,
                    置信率 REAL,
                    推理耗时 REAL,
                    图片路径 VARCHAR(256),
                    是否正确 VARCHAR(256)
                )
            ''')
            print("SQLite数据库表已创建")
        
        conn.commit()
        conn.close()
    
    def _init_mysql_database(self):
        """初始化MySQL数据库"""
        conn = pymysql.connect(**self.mysql_config)
        cursor = conn.cursor()
        
        # 创建数据库（如果不存在）
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.mysql_config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.commit()
        except Exception as e:
            print(f"创建数据库失败: {e}")
        
        # 重新连接到指定数据库
        conn.close()
        conn = pymysql.connect(**self.mysql_config)
        cursor = conn.cursor()
        
        # 检查表是否已存在
        cursor.execute("SHOW TABLES LIKE 'inference_results'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            # 创建推理结果表
            cursor.execute('''
                CREATE TABLE inference_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    推理时间 DATETIME NOT NULL,
                    图片名称 VARCHAR(256) NOT NULL,
                    AI判定结果 VARCHAR(256) NOT NULL,
                    置信率 FLOAT,
                    推理耗时 FLOAT,
                    图片路径 VARCHAR(256),
                    是否正确 VARCHAR(256)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''')
            print("MySQL数据库表已创建")
        else:
            print("MySQL数据库表已存在")
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """获取数据库连接"""
        if self.db_type == "mysql":
            return pymysql.connect(**self.mysql_config)
        else:
            return sqlite3.connect(str(self.db_file))
    
    def add_result_to_history(self, result: Dict, true_label: str = None) -> bool:
        """
        添加单条结果到历史记录
        
        Args:
            result: 推理结果字典，包含：
                - image_path: 图片路径
                - class: 预测类别
                - confidence: 置信度
                - inference_time: 推理耗时
                - timestamp: 推理时间戳
            true_label: 真实标签（可选）
            
        Returns:
            bool: 是否添加成功
        """
        try:
            image_name = Path(result.get('image_path', '')).name
            pred_class = result.get('class', 'N/A')
            confidence = result.get('confidence', 0)
            inference_time = result.get('inference_time', 0)
            timestamp = result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            image_path = result.get('image_path', '')
            
            # 计算是否正确
            is_correct = 'N/A'
            if true_label and pred_class != 'N/A':
                is_correct = '是' if pred_class == true_label else '否'
            
            # 保存到CSV
            with open(self.history_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    image_name,
                    pred_class,
                    f'{confidence:.4f}',
                    f'{inference_time:.4f}',
                    image_path,
                    is_correct
                ])
            
            # 保存到数据库
            self._save_to_database(timestamp, image_name, pred_class, confidence, 
                                 inference_time, image_path, is_correct)
            
            return True
        except Exception as e:
            print(f"添加结果到历史失败: {e}")
            return False
    
    def _save_to_database(self, timestamp, image_name, predicted_class, confidence, 
                         inference_time, image_path, is_correct):
        """保存结果到数据库"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if self.db_type == "mysql":
                # MySQL使用%s作为占位符
                cursor.execute('''
                    INSERT INTO inference_results 
                    (推理时间, 图片名称, AI判定结果, 置信率, 推理耗时, 
                     图片路径, 是否正确)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (timestamp, image_name, predicted_class, confidence, inference_time,
                      image_path, is_correct))
            else:
                # SQLite使用?作为占位符
                cursor.execute('''
                    INSERT INTO inference_results 
                    (推理时间, 图片名称, AI判定结果, 置信率, 推理耗时, 
                     图片路径, 是否正确)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, image_name, predicted_class, confidence, inference_time,
                      image_path, is_correct))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"保存到数据库失败: {e}")
            return False
    
    def load_results_from_database(self, limit: int = None) -> List[Dict]:
        """
        从数据库加载推理结果
        
        Args:
            limit: 限制返回的记录数（可选）
            
        Returns:
            List[Dict]: 推理结果列表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if limit:
                if self.db_type == "mysql":
                    cursor.execute('SELECT * FROM inference_results ORDER BY 推理时间 DESC LIMIT %s', (limit,))
                else:
                    cursor.execute('SELECT * FROM inference_results ORDER BY 推理时间 DESC LIMIT ?', (limit,))
            else:
                cursor.execute('SELECT * FROM inference_results ORDER BY 推理时间 DESC')
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                result = {
                    'id': row[0],
                    'timestamp': row[1],
                    'image_name': row[2],
                    'class': row[3],
                    'confidence': row[4],
                    'inference_time': row[5],
                    'image_path': row[6],
                    'is_correct': row[7]
                }
                results.append(result)
            
            return results
        except Exception as e:
            print(f"从数据库加载结果失败: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute('SELECT COUNT(*) FROM inference_results')
            total_count = cursor.fetchone()[0]
            
            # 各类别统计
            cursor.execute('SELECT 预测类别, COUNT(*) FROM inference_results GROUP BY 预测类别')
            class_stats = dict(cursor.fetchall())
            
            # 平均置信度
            cursor.execute('SELECT AVG(置信度) FROM inference_results WHERE 置信度 IS NOT NULL')
            avg_confidence = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_records': total_count,
                'class_distribution': class_stats,
                'average_confidence': avg_confidence
            }
        except Exception as e:
            print(f"获取数据库统计失败: {e}")
            return {'total_records': 0, 'class_distribution': {}, 'average_confidence': 0}
    
    def batch_add_results_to_history(self, results: List[Dict], true_labels: List[str] = None) -> int:
        """
        批量添加结果到历史记录
        
        Args:
            results: 推理结果列表
            true_labels: 真实标签列表（可选，与results对应）
            
        Returns:
            int: 成功添加的结果数
        """
        success_count = 0
        for idx, result in enumerate(results):
            true_label = None
            if true_labels and idx < len(true_labels):
                true_label = true_labels[idx]
            
            if self.add_result_to_history(result, true_label):
                success_count += 1
        
        return success_count
    
    def export_to_csv(self, results: List[Dict], true_labels: List[str] = None, 
                     custom_filename: str = None) -> str:
        """
        导出批量推理结果为CSV文件
        
        Args:
            results: 推理结果列表
            true_labels: 真实标签列表（可选）
            custom_filename: 自定义文件名（可选）
            
        Returns:
            str: 导出文件的完整路径
        """
        try:
            if custom_filename:
                csv_filename = self.output_folder / custom_filename
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = self.output_folder / f"inference_results_{timestamp}.csv"
            
            with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)
                
                for idx, result in enumerate(results):
                    image_name = Path(result.get('image_path', '')).name
                    pred_class = result.get('class', 'N/A')
                    confidence = result.get('confidence', 0)
                    inference_time = result.get('inference_time', 0)
                    timestamp = result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    image_path = result.get('image_path', '')
                    
                    is_correct = 'N/A'
                    if true_labels and idx < len(true_labels):
                        true_label = true_labels[idx]
                        if pred_class != 'N/A':
                            is_correct = '是' if pred_class == true_label else '否'
                    
                    writer.writerow([
                        timestamp,
                        image_name,
                        pred_class,
                        f'{confidence:.4f}',
                        f'{inference_time:.4f}',
                        image_path,
                        is_correct
                    ])
            
            return str(csv_filename)
        except Exception as e:
            raise Exception(f"导出CSV失败: {e}")
    
    def calculate_metrics(self, results: List[Dict], true_labels: List[str] = None) -> Dict:
        """
        计算精度和召回率等指标
        
        Args:
            results: 推理结果列表
            true_labels: 真实标签列表
            
        Returns:
            dict: 包含各项指标的字典
                {
                    'total': 总数,
                    'correct': 正确数,
                    'accuracy': 准确率,
                    'precision': 精准率（各类别）,
                    'recall': 召回率（各类别）,
                    'f1_score': F1分数（各类别）,
                    'class_wise_metrics': 各类别的详细指标
                }
        """
        if not true_labels or len(true_labels) != len(results):
            return {
                'total': len(results),
                'correct': 0,
                'accuracy': 0.0,
                'precision': {},
                'recall': {},
                'f1_score': {},
                'error_message': '缺少真实标签或标签数量不匹配'
            }
        
        metrics = {
            'total': len(results),
            'correct': 0,
            'accuracy': 0.0,
            'precision': {},
            'recall': {},
            'f1_score': {},
            'class_wise_metrics': {}
        }
        
        # 获取所有类别
        all_classes = set()
        for result in results:
            if 'class' in result and result['class'] != 'N/A':
                all_classes.add(result['class'])
        for label in true_labels:
            if label != 'N/A':
                all_classes.add(label)
        all_classes = sorted(list(all_classes))
        
        # 初始化各类别统计
        class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_classes}
        
        # 计算混淆矩阵
        for idx, result in enumerate(results):
            pred_class = result.get('class', 'N/A')
            true_class = true_labels[idx]
            
            if pred_class != 'N/A' and true_class != 'N/A':
                if pred_class == true_class:
                    metrics['correct'] += 1
                    class_stats[true_class]['tp'] += 1
                else:
                    class_stats[pred_class]['fp'] += 1
                    class_stats[true_class]['fn'] += 1
        
        # 计算准确率
        metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
        
        # 计算各类别的精准率、召回率和F1分数
        for cls in all_classes:
            tp = class_stats[cls]['tp']
            fp = class_stats[cls]['fp']
            fn = class_stats[cls]['fn']
            
            # 精准率 = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['precision'][cls] = precision
            
            # 召回率 = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['recall'][cls] = recall
            
            # F1分数 = 2 * (精准率 * 召回率) / (精准率 + 召回率)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            metrics['f1_score'][cls] = f1
            
            metrics['class_wise_metrics'][cls] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # 计算加权平均精准率
        total_predicted = sum(stats['tp'] + stats['fp'] for stats in class_stats.values())
        if total_predicted > 0:
            weighted_precision = sum(
                metrics['precision'].get(cls, 0) * (class_stats[cls]['tp'] + class_stats[cls]['fp'])
                for cls in all_classes
            ) / total_predicted
        else:
            weighted_precision = 0.0
        metrics['weighted_precision'] = weighted_precision
        
        return metrics
    
    def get_statistics_text(self, results: List[Dict], true_labels: List[str] = None) -> str:
        """
        生成统计信息文本
        
        Args:
            results: 推理结果列表
            true_labels: 真实标签列表（可选）
            
        Returns:
            str: 统计信息文本
        """
        stats_text = f"\n推理统计信息\n{'='*60}\n"
        stats_text += f"推理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_text += f"总推理图片数: {len(results)}\n"
        
        # 计算基本统计
        class_counts = {}
        total_confidence = 0
        valid_count = 0
        total_time = 0
        
        for result in results:
            if result.get('class') != 'N/A' and result.get('class') != '错误':
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_confidence += result.get('confidence', 0)
                total_time += result.get('inference_time', 0)
                valid_count += 1
        
        stats_text += f"成功推理: {valid_count}\n"
        stats_text += f"失败: {len(results) - valid_count}\n"
        
        if valid_count > 0:
            stats_text += f"平均置信度: {total_confidence / valid_count:.4f}\n"
            stats_text += f"总耗时: {total_time:.4f}秒\n"
            stats_text += f"平均耗时: {total_time / valid_count:.4f}秒/张\n"
        
        stats_text += f"\n各类别统计:\n"
        for class_name, count in sorted(class_counts.items()):
            percentage = count / valid_count * 100 if valid_count > 0 else 0
            stats_text += f"  {class_name}: {count} ({percentage:.1f}%)\n"
        
        # 如果有真实标签，计算精度指标
        if true_labels:
            metrics = self.calculate_metrics(results, true_labels)
            stats_text += f"\n精度指标:\n"
            stats_text += f"  总体准确率 (Accuracy): {metrics['accuracy']:.4f}\n"
            stats_text += f"  加权平均精准率 (Weighted Precision): {metrics.get('weighted_precision', 0):.4f}\n"
            
            stats_text += f"\n各类别详细指标:\n"
            for cls in sorted(metrics['class_wise_metrics'].keys()):
                cls_metrics = metrics['class_wise_metrics'][cls]
                stats_text += f"  {cls}:\n"
                stats_text += f"    TP: {cls_metrics['tp']}, FP: {cls_metrics['fp']}, FN: {cls_metrics['fn']}\n"
                stats_text += f"    精准率 (Precision): {cls_metrics['precision']:.4f}\n"
                stats_text += f"    召回率 (Recall): {cls_metrics['recall']:.4f}\n"
                stats_text += f"    F1分数: {cls_metrics['f1_score']:.4f}\n"
        
        return stats_text
    
    def load_history(self, limit: int = None) -> List[Dict]:
        """
        加载历史记录
        
        Args:
            limit: 限制记录数（None表示加载全部）
            
        Returns:
            list: 历史记录列表
        """
        try:
            if HAS_PANDAS:
                df = pd.read_csv(self.history_file, encoding='utf-8-sig')
                if limit:
                    df = df.tail(limit)
                return df.to_dict('records')
            else:
                # 没有pandas时的备选方案
                records = []
                with open(self.history_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(row)
                
                if limit and len(records) > limit:
                    records = records[-limit:]
                return records
        except Exception as e:
            print(f"加载历史失败: {e}")
            return []
    
    def get_summary_stats(self) -> Dict:
        """
        获取历史记录的汇总统计
        
        Returns:
            dict: 汇总统计信息
        """
        try:
            if HAS_PANDAS:
                df = pd.read_csv(self.history_file, encoding='utf-8-sig')
            else:
                # 没有pandas时手动读取
                records = self.load_history()
                if not records:
                    return {'error': '历史记录为空'}
                
                # 手动计算统计
                summary = {
                    'total_inferences': len(records),
                    'successful': len([r for r in records if r.get('AI判定结果', r.get('预测类别')) != 'N/A']),
                    'class_distribution': {}
                }
                
                for record in records:
                    cls = record.get('AI判定结果', record.get('预测类别', 'N/A'))
                    if cls != 'N/A':
                        summary['class_distribution'][cls] = summary['class_distribution'].get(cls, 0) + 1
                
                # 计算平均置信度和总耗时
                confidences = []
                times = []
                for record in records:
                    try:
                        conf = float(record.get('置信率', record.get('置信度', 0)))
                        confidences.append(conf)
                    except (ValueError, TypeError):
                        pass
                    try:
                        t = float(record.get('推理耗时(秒)', 0))
                        times.append(t)
                    except (ValueError, TypeError):
                        pass
                
                if confidences:
                    summary['avg_confidence'] = sum(confidences) / len(confidences)
                if times:
                    summary['total_time'] = sum(times)
                
                # 计算精度
                if '是否正确' in (records[0] if records else {}):
                    correct_count = len([r for r in records if r.get('是否正确') == '是'])
                    evaluated_count = len([r for r in records if r.get('是否正确') != 'N/A'])
                    if evaluated_count > 0:
                        summary['accuracy'] = correct_count / evaluated_count
                        summary['correct_count'] = correct_count
                        summary['evaluated_count'] = evaluated_count
                
                return summary
            
            if df.empty:
                return {'error': '历史记录为空'}
            
            # 计算统计信息
            prediction_col = 'AI判定结果' if 'AI判定结果' in df.columns else '预测类别'
            confidence_col = '置信率' if '置信率' in df.columns else '置信度'
            summary = {
                'total_inferences': len(df),
                'successful': len(df[df[prediction_col] != 'N/A']),
                'class_distribution': df[prediction_col].value_counts().to_dict(),
                'avg_confidence': float(df[confidence_col].astype(float).mean()),
                'total_time': float(df['推理耗时(秒)'].astype(float).sum()),
            }
            
            # 计算精度（如果有正确性信息）
            if '是否正确' in df.columns:
                correct_count = len(df[df['是否正确'] == '是'])
                evaluated_count = len(df[df['是否正确'] != 'N/A'])
                if evaluated_count > 0:
                    summary['accuracy'] = correct_count / evaluated_count
                    summary['correct_count'] = correct_count
                    summary['evaluated_count'] = evaluated_count
            
            return summary
        except Exception as e:
            return {'error': str(e)}
