"""
PyQt5 推理UI应用 - 图形化界面实现
"""
import sys
import os
from pathlib import Path
import cv2
from datetime import datetime
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QScrollArea, QSplitter, QProgressBar, QMessageBox, QComboBox,
    QSpinBox, QGroupBox, QGridLayout, QListWidget, QListWidgetItem,
    QTabWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDir, QFileSystemWatcher
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
from PyQt5.QtGui import QPainter, QColor as QtColor
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.inference_engine import InferenceEngine
from src.utils.result_manager import ResultManager
# 禁止albumentations的更新检查弹窗
os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"

class InferenceWorker(QThread):
    """推理工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, inference_engine, image_paths):
        super().__init__()
        self.inference_engine = inference_engine
        self.image_paths = image_paths
    
    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            for idx, image_path in enumerate(self.image_paths):
                try:
                    start_time = time.time()
                    result = self.inference_engine.infer(image_path)
                    elapsed = time.time() - start_time
                    result['image_path'] = str(image_path)
                    result['inference_time'] = elapsed
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    results.append(result)
                except Exception as e:
                    results.append({
                        'image_path': str(image_path),
                        'error': str(e),
                        'class': '错误',
                        'inference_time': 0,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                progress_percent = int((idx + 1) / total * 100)
                self.progress.emit(progress_percent)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class AutoInferenceWorker(QThread):
    """自动推理并移动图片的工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, inference_engine, image_paths, ai_result_folder):
        super().__init__()
        self.inference_engine = inference_engine
        self.image_paths = image_paths
        self.ai_result_folder = ai_result_folder
    
    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            for idx, image_path in enumerate(self.image_paths):
                try:
                    start_time = time.time()
                    result = self.inference_engine.infer(image_path)
                    elapsed = time.time() - start_time
                    result['image_path'] = str(image_path)
                    result['inference_time'] = elapsed
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    results.append(result)
                    # 根据分类移动图片
                    cls = result.get('class', 'NG')
                    dest_dir = Path(self.ai_result_folder) / cls
                    dest_dir.mkdir(exist_ok=True, parents=True)
                    shutil.move(str(image_path), str(dest_dir / Path(image_path).name))
                except Exception as e:
                    results.append({
                        'image_path': str(image_path),
                        'error': str(e),
                        'class': '错误',
                        'inference_time': 0,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                progress_percent = int((idx + 1) / total * 100)
                self.progress.emit(progress_percent)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ContinuousAutoInferenceWorker(QThread):
    """持续自动推理并移动图片的工作线程"""
    progress = pyqtSignal(int)
    result_processed = pyqtSignal(dict)  # 每处理一张图片时发出信号
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, inference_engine, inference_folder, ai_result_folder, scan_interval=2):
        super().__init__()
        self.inference_engine = inference_engine
        self.inference_folder = Path(inference_folder)
        self.ai_result_folder = Path(ai_result_folder)
        self.scan_interval = scan_interval
        self.running = True
        self.processed_images = set()  # 已处理的图片集合
        
    def stop(self):
        """停止持续扫描"""
        self.running = False
    
    def run(self):
        """持续扫描并推理"""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
            
            self.status_changed.emit(f"持续扫描已开启，监控文件夹: {self.inference_folder}")
            
            while self.running:
                try:
                    # 获取当前文件夹中的所有图片
                    current_images = {
                        p for p in self.inference_folder.iterdir() 
                        if p.suffix in image_extensions and p.is_file()
                    }
                    
                    # 找出新增的图片（未被处理过的）
                    new_images = current_images - self.processed_images
                    
                    if new_images:
                        self.status_changed.emit(f"发现 {len(new_images)} 张新图片，开始推理...")
                        
                        for image_path in sorted(new_images):
                            if not self.running:
                                break
                                
                            try:
                                start_time = time.time()
                                result = self.inference_engine.infer(str(image_path))
                                elapsed = time.time() - start_time
                                result['image_path'] = str(image_path)
                                result['inference_time'] = elapsed
                                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                
                                # 根据分类移动图片
                                cls = result.get('class', 'NG')
                                dest_dir = self.ai_result_folder / cls
                                dest_dir.mkdir(exist_ok=True, parents=True)
                                
                                # 移动文件
                                dest_path = dest_dir / image_path.name
                                shutil.move(str(image_path), str(dest_path))
                                
                                # 标记为已处理
                                self.processed_images.add(image_path)
                                
                                # 发出结果信号
                                self.result_processed.emit(result)
                                
                                self.status_changed.emit(
                                    f"已处理: {image_path.name} -> {cls} "
                                    f"(置信度: {result.get('confidence', 0):.2%})"
                                )
                                
                            except Exception as e:
                                error_result = {
                                    'image_path': str(image_path),
                                    'error': str(e),
                                    'class': '错误',
                                    'confidence': 0,
                                    'inference_time': 0,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                self.result_processed.emit(error_result)
                                self.status_changed.emit(f"处理失败: {image_path.name} - {str(e)}")
                    else:
                        self.status_changed.emit(f"等待新图片... ({datetime.now().strftime('%H:%M:%S')})")
                    
                    # 等待指定的扫描间隔
                    time.sleep(self.scan_interval)
                    
                except Exception as e:
                    self.status_changed.emit(f"扫描错误: {str(e)}")
                    time.sleep(self.scan_interval)
            
            self.status_changed.emit("持续扫描已停止")
            
        except Exception as e:
            self.error.emit(f"持续扫描过程中出错: {str(e)}")


class InferenceUI(QMainWindow):
    """推理UI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.inference_engine = None
        self.current_results = []
        self.all_results = []  # 保留所有历史推理结果
        self.inference_folder = Path("Inference")
        self.output_folder = Path("Inference/results")
        self.ai_result_folder = Path("AIResult")
        
        # 持续扫描相关
        self.continuous_worker = None
        self.is_scanning = False
        self.scan_results = []  # 持续扫描的结果列表
        
        # 初始化结果管理器
        self.result_manager = ResultManager(str(self.output_folder))
        
        # 创建必要的文件夹
        self.inference_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        # AIResult目录及子目录
        self.ai_result_folder.mkdir(exist_ok=True)
        (self.ai_result_folder / "OK").mkdir(exist_ok=True, parents=True)
        (self.ai_result_folder / "NG").mkdir(exist_ok=True, parents=True)
        
        self.init_ui()
        self.load_inference_images()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("金手指不良判断系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置字体
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        
        # 创建中心widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 1. 配置区域
        config_layout = self._create_config_layout()
        main_layout.addWidget(config_layout)
        
        # 2. 创建选项卡
        tab_widget = QTabWidget()
        
        # 选项卡1: 图片推理
        inference_tab = self._create_inference_tab()
        tab_widget.addTab(inference_tab, "图片推理")
        
        # 选项卡2: 批量推理记录
        batch_tab = self._create_batch_tab()
        tab_widget.addTab(batch_tab, "推理结果记录")
        
        # 选项卡3: 统计信息
        stats_tab = self._create_stats_tab()
        tab_widget.addTab(stats_tab, "统计信息")
        
        main_layout.addWidget(tab_widget)
        
        central_widget.setLayout(main_layout)
    
    def _create_config_layout(self):
        """创建配置区域"""
        config_group = QGroupBox("配置")
        config_layout = QGridLayout()
        
        # 模型选择
        config_layout.addWidget(QLabel("选择模型:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Swin_V2_B', 'Swin_B', "ResNeXt50_32X4D",'resneXt101(32x8d)',"ResNeXt101_64X4D"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        config_layout.addWidget(self.model_combo, 0, 1)
        
        # 图像大小
        config_layout.addWidget(QLabel("图像大小:"), 0, 2)
        self.size_spin = QSpinBox()
        self.size_spin.setMinimum(64)
        self.size_spin.setMaximum(512)
        self.size_spin.setValue(224)
        config_layout.addWidget(self.size_spin, 0, 3)
        
        # 推理文件夹路径显示
        config_layout.addWidget(QLabel("推理文件夹:"), 1, 0)
        self.folder_label = QLabel(str(self.inference_folder.absolute()))
        self.folder_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        config_layout.addWidget(self.folder_label, 1, 1, 1, 2)
        
        # 打开文件夹按钮
        open_folder_btn = QPushButton("打开文件夹")
        open_folder_btn.clicked.connect(self.open_inference_folder)
        config_layout.addWidget(open_folder_btn, 1, 3)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新图片列表")
        refresh_btn.clicked.connect(self.load_inference_images)
        config_layout.addWidget(refresh_btn, 2, 0)
        
        # 初始化模型按钮
        init_btn = QPushButton("初始化模型")
        init_btn.clicked.connect(self.init_model)
        config_layout.addWidget(init_btn, 2, 1)
        
        # 状态标签
        self.status_label = QLabel("状态: 未初始化")
        self.status_label.setStyleSheet("color: red;")
        config_layout.addWidget(self.status_label, 2, 2, 1, 2)
        
        config_group.setLayout(config_layout)
        return config_group
    
    def _create_inference_tab(self):
        """创建推理选项卡"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # 左侧: 图片列表
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Inference文件夹中的图片:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        left_layout.addWidget(self.image_list)
        
        # 推理按钮
        infer_btn = QPushButton("推理选中图片")
        infer_btn.clicked.connect(self.infer_selected_image)
        left_layout.addWidget(infer_btn)
        
        # 批量推理按钮
        batch_infer_btn = QPushButton("批量推理所有图片")
        batch_infer_btn.clicked.connect(self.batch_infer_all)
        left_layout.addWidget(batch_infer_btn)
        
        # 自动推理并移动按钮
        auto_infer_btn = QPushButton("自动推理并移动")
        auto_infer_btn.clicked.connect(self.auto_infer_and_move)
        left_layout.addWidget(auto_infer_btn)
        
        # 持续扫描按钮
        self.continuous_scan_btn = QPushButton("开启持续扫描")
        self.continuous_scan_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.continuous_scan_btn.clicked.connect(self.toggle_continuous_scan)
        left_layout.addWidget(self.continuous_scan_btn)
        
        # 持续扫描状态标签
        self.scan_status_label = QLabel("扫描状态: 未启动")
        self.scan_status_label.setStyleSheet("color: gray;")
        left_layout.addWidget(self.scan_status_label)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(250)
        
        # 右侧: 图片预览和推理结果
        right_layout = QVBoxLayout()
        
        # 图片预览
        right_layout.addWidget(QLabel("图片预览:"))
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.image_preview.setMinimumHeight(300)
        right_layout.addWidget(self.image_preview)
        
        # 推理结果
        right_layout.addWidget(QLabel("推理结果:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        right_layout.addWidget(self.result_text)
        
        # 保存结果按钮
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self.save_result)
        right_layout.addWidget(save_btn)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        tab.setLayout(layout)
        return tab
    
    def _create_batch_tab(self):
        """创建批量推理结果选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("批量推理结果:"))
        
        # 结果表格
        self.result_table = QTableWidget()
        # 5列: 图片名称、预测类别、置信度、耗时、完成时间
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['图片名称', '预测类别', '置信度', '耗时', '完成时间'])
        self.result_table.setColumnWidth(0, 250)
        self.result_table.setColumnWidth(1, 100)
        self.result_table.setColumnWidth(2, 100)
        self.result_table.setColumnWidth(3, 100)
        self.result_table.setColumnWidth(4, 150)
        layout.addWidget(self.result_table)
        
        # 导出结果按钮
        export_btn = QPushButton("导出结果为CSV")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)
        
        tab.setLayout(layout)
        return tab
    
    def _create_stats_tab(self):
        """创建统计信息选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("推理结果统计:"))
        
        # 统计信息展示
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        tab.setLayout(layout)
        return tab
    
    def on_model_changed(self):
        """模型变更回调，自动初始化新模型"""
        self.status_label.setText("状态: 模型已改变，正在重新初始化...")
        self.status_label.setStyleSheet("color: orange;")
        self.inference_engine = None
        # 尝试直接初始化新模型
        # 使用定时器以便界面有机会刷新状态标签
        QTimer.singleShot(100, self.init_model)
    
    def init_model(self):
        """初始化模型"""
        try:
            model_name = self.model_combo.currentText()
            image_size = (self.size_spin.value(), self.size_spin.value())
            checkpoint = f'experiments/{model_name}/checkpoints/best_{model_name}_model.pth'
            
            if not os.path.exists(checkpoint):
                QMessageBox.critical(self, "错误", f"找不到模型文件: {checkpoint}")
                return
            
            self.status_label.setText("正在初始化模型...")
            QApplication.processEvents()
            
            self.inference_engine = InferenceEngine(model_name, checkpoint, image_size)
            
            self.status_label.setText(f"状态: 模型已初始化 ({model_name})")
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "成功", f"模型 {model_name} 初始化成功！")
            
        except Exception as e:
            self.status_label.setText(f"状态: 初始化失败")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")
    
    def load_inference_images(self):
        """加载Inference文件夹中的图片"""
        self.image_list.clear()
        
        if not self.inference_folder.exists():
            self.inference_folder.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_paths = [p for p in self.inference_folder.iterdir() 
                      if p.suffix in image_extensions]
        
        for image_path in sorted(image_paths):
            item = QListWidgetItem(image_path.name)
            item.setData(Qt.UserRole, str(image_path))
            self.image_list.addItem(item)
        
        status = f"找到 {len(image_paths)} 张图片"
        if len(image_paths) == 0:
            status += f"\n请将图片放入: {self.inference_folder.absolute()}"
    
    def on_image_selected(self, item):
        """图片选中回调"""
        image_path = item.data(Qt.UserRole)
        self.display_image_preview(image_path)
    
    def display_image_preview(self, image_path):
        """显示图片预览"""
        try:
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
            self.image_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self.image_preview.setText(f"无法加载图片: {str(e)}")
    
    def infer_selected_image(self):
        """推理选中的图片"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return
        
        current_item = self.image_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "警告", "请选择一张图片")
            return
        
        try:
            image_path = current_item.data(Qt.UserRole)
            result = self.inference_engine.infer(image_path)
            self.display_inference_result(result)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败: {str(e)}")
    
    def display_inference_result(self, result):
        """显示推理结果"""
        result_text = f"""
推理结果:
{'='*50}
预测类别: {result['class']}
置信度: {result['confidence']:.2%}

各类别概率:
"""
        for class_name, prob in result['probabilities'].items():
            result_text += f"  {class_name}: {prob:.2%}\n"
        
        self.result_text.setText(result_text)
    
    def batch_infer_all(self):
        """批量推理所有图片"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return
        
        if self.image_list.count() == 0:
            QMessageBox.warning(self, "警告", "没有找到图片")
            return
        
        image_paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.UserRole)
            image_paths.append(image_path)
        
        # 启动推理线程
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = InferenceWorker(self.inference_engine, image_paths)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_batch_infer_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "错误", f"批量推理失败: {e}"))
        self.worker.start()

    def auto_infer_and_move(self):
        """自动推理Inference目录中的所有图片并根据结果移动到AIResult/OK或NG"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return
        
        # 获取所有图片路径
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_paths = [str(p) for p in self.inference_folder.iterdir() if p.suffix in image_extensions]
        if not image_paths:
            QMessageBox.warning(self, "警告", "Inference文件夹中没有图片可供推理")
            return
        
        # 启动自动推理线程
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = AutoInferenceWorker(self.inference_engine, image_paths, self.ai_result_folder)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_auto_infer_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "错误", f"自动推理失败: {e}"))
        self.worker.start()

    def toggle_continuous_scan(self):
        """切换持续扫描状态"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return
        
        if self.is_scanning:
            # 停止扫描
            self.stop_continuous_scan()
        else:
            # 开启扫描
            self.start_continuous_scan()
    
    def start_continuous_scan(self):
        """开启持续扫描"""
        self.is_scanning = True
        self.scan_results = []
        
        # 启动持续扫描线程
        self.continuous_worker = ContinuousAutoInferenceWorker(
            self.inference_engine, 
            self.inference_folder,
            self.ai_result_folder,
            scan_interval=2  # 每2秒扫描一次
        )
        self.continuous_worker.result_processed.connect(self.on_continuous_result)
        self.continuous_worker.status_changed.connect(self.on_scan_status_changed)
        self.continuous_worker.error.connect(self.on_scan_error)
        self.continuous_worker.start()
        
        # 更新按钮样式
        self.continuous_scan_btn.setText("关闭持续扫描")
        self.continuous_scan_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        
        self.scan_status_label.setStyleSheet("color: green;")
        self.scan_status_label.setText("扫描状态: 运行中...")
    
    def stop_continuous_scan(self):
        """停止持续扫描"""
        if self.continuous_worker is not None:
            self.continuous_worker.stop()
            self.continuous_worker.wait()  # 等待线程结束
        
        self.is_scanning = False
        
        # 更新按钮样式
        self.continuous_scan_btn.setText("开启持续扫描")
        self.continuous_scan_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.scan_status_label.setStyleSheet("color: gray;")
        self.scan_status_label.setText("扫描状态: 已停止")
        
        # 弹出完成提示
        if self.scan_results:
            QMessageBox.information(
                self, 
                "持续扫描已停止", 
                f"本次扫描共处理 {len(self.scan_results)} 张图片"
            )
    
    def on_continuous_result(self, result):
        """处理持续扫描的单个结果"""
        self.scan_results.append(result)
        
        # 更新结果表格
        row_count = self.result_table.rowCount()
        self.result_table.insertRow(row_count)
        
        image_name = Path(result['image_path']).name
        pred_class = result.get('class', '错误')
        confidence = result.get('confidence', 0)
        inference_time = result.get('inference_time', 0)
        timestamp = result.get('timestamp', '')
        
        self.result_table.setItem(row_count, 0, QTableWidgetItem(image_name))
        
        class_item = QTableWidgetItem(pred_class)
        if pred_class != '错误':
            if pred_class == 'OK':
                class_item.setBackground(QColor(144, 238, 144))  # 浅绿色
            else:
                class_item.setBackground(QColor(255, 182, 193))  # 浅红色
        self.result_table.setItem(row_count, 1, class_item)
        
        conf_text = f"{confidence:.2%}" if confidence else "N/A"
        self.result_table.setItem(row_count, 2, QTableWidgetItem(conf_text))
        
        time_text = f"{inference_time:.3f}s" if inference_time else "N/A"
        self.result_table.setItem(row_count, 3, QTableWidgetItem(time_text))
        
        self.result_table.setItem(row_count, 4, QTableWidgetItem(timestamp))
        
        # 滚动到最新的行
        self.result_table.scrollToBottom()
    
    def on_scan_status_changed(self, status):
        """更新扫描状态信息"""
        self.scan_status_label.setText(f"扫描状态: {status}")
        # 保持标签为绿色（运行中）
        if self.is_scanning:
            self.scan_status_label.setStyleSheet("color: green;")
    
    def on_scan_error(self, error):
        """扫描错误处理"""
        self.scan_status_label.setText(f"扫描状态: 错误 - {error}")
        self.scan_status_label.setStyleSheet("color: red;")
        QMessageBox.critical(self, "扫描错误", f"持续扫描发生错误: {error}")

    def on_auto_infer_finished(self, results):
        """自动推理完成回调"""
        self.progress_bar.setVisible(False)
        self.current_results = results
        
        # 将结果添加到历史记录
        self.result_manager.batch_add_results_to_history(results)
        
        # 更新批量结果表格，与批量推理一致
        self.result_table.setRowCount(len(results))
        for row, result in enumerate(results):
            image_name = Path(result['image_path']).name
            pred_class = result.get('class', '错误')
            confidence = result.get('confidence', 0)
            inference_time = result.get('inference_time', 0)
            timestamp = result.get('timestamp', '')
            
            self.result_table.setItem(row, 0, QTableWidgetItem(image_name))
            class_item = QTableWidgetItem(pred_class)
            if pred_class != '错误':
                if pred_class == 'OK':
                    class_item.setBackground(QColor(144, 238, 144))
                else:
                    class_item.setBackground(QColor(255, 182, 193))
            self.result_table.setItem(row, 1, class_item)
            conf_text = f"{confidence:.2%}" if confidence else "N/A"
            self.result_table.setItem(row, 2, QTableWidgetItem(conf_text))
            time_text = f"{inference_time:.3f}s" if inference_time else "N/A"
            self.result_table.setItem(row, 3, QTableWidgetItem(time_text))
            self.result_table.setItem(row, 4, QTableWidgetItem(timestamp))
        # 移动完成后刷新列表
        self.load_inference_images()
        # 更新统计信息
        self.update_statistics()
        QMessageBox.information(self, "完成", f"自动推理并移动完成，共处理 {len(results)} 张图片")
    
    def on_batch_infer_finished(self, results):
        """批量推理完成回调"""
        self.progress_bar.setVisible(False)
        self.current_results = results
        
        # 将结果添加到历史记录
        self.result_manager.batch_add_results_to_history(results)
        
        # 更新结果表格
        self.result_table.setRowCount(len(results))
        
        for row, result in enumerate(results):
            image_name = Path(result['image_path']).name
            pred_class = result.get('class', '错误')
            confidence = result.get('confidence', 0)
            inference_time = result.get('inference_time', 0)
            timestamp = result.get('timestamp', '')
            
            # 图片名称
            self.result_table.setItem(row, 0, QTableWidgetItem(image_name))
            
            # 预测类别
            class_item = QTableWidgetItem(pred_class)
            if pred_class != '错误':
                if pred_class == 'OK':
                    class_item.setBackground(QColor(144, 238, 144))  # 浅绿色
                else:
                    class_item.setBackground(QColor(255, 182, 193))  # 浅红色
            self.result_table.setItem(row, 1, class_item)
            
            # 置信度
            conf_text = f"{confidence:.2%}" if confidence else "N/A"
            self.result_table.setItem(row, 2, QTableWidgetItem(conf_text))
            
            # 推理耗时
            time_text = f"{inference_time:.3f}s" if inference_time else "N/A"
            self.result_table.setItem(row, 3, QTableWidgetItem(time_text))
            
            # 完成时间
            self.result_table.setItem(row, 4, QTableWidgetItem(timestamp))
        
        # 更新统计信息
        self.update_statistics()
        
        QMessageBox.information(self, "成功", f"批量推理完成！共推理 {len(results)} 张图片")
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.current_results:
            return
        
        # 从文件名自动提取真实标签（用于计算准确率）
        try:
            from tools.result_analyzer import ResultAnalyzer
            true_labels = [
                ResultAnalyzer.extract_true_label_from_filename(Path(r['image_path']).name)
                for r in self.current_results
            ]
            # 使用结果管理器生成统计信息（包含准确率）
            stats_text = self.result_manager.get_statistics_text(self.current_results, true_labels)
        except (ImportError, AttributeError):
            # 如果无法导入或提取标签，则显示不含准确率的统计信息
            stats_text = self.result_manager.get_statistics_text(self.current_results)
        
        self.stats_text.setText(stats_text)
    
    def save_result(self):
        """保存单张图片的推理结果"""
        if not self.result_text.toPlainText():
            QMessageBox.warning(self, "警告", "没有推理结果可保存")
            return
        
        current_item = self.image_list.currentItem()
        if current_item is None:
            return
        
        image_name = current_item.text()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{self.output_folder}/{image_name.split('.')[0]}_result_{timestamp}.txt"
        
        try:
            with open(result_filename, 'w', encoding='utf-8-sig') as f:
                f.write(f"图片: {image_name}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(self.result_text.toPlainText())
            
            QMessageBox.information(self, "成功", f"结果已保存: {result_filename}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def export_results(self):
        """导出批量推理结果"""
        if not self.current_results:
            QMessageBox.warning(self, "警告", "没有推理结果可导出")
            return
        
        try:
            # 使用结果管理器导出CSV
            csv_path = self.result_manager.export_to_csv(self.current_results)
            QMessageBox.information(self, "成功", f"结果已导出: {csv_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def open_inference_folder(self):
        """打开推理文件夹"""
        import subprocess
        folder_path = self.inference_folder.absolute()
        
        try:
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', folder_path])
            else:  # Linux
                subprocess.Popen(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件夹: {str(e)}")
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 确保持续扫描被停止
        if self.is_scanning:
            self.stop_continuous_scan()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = InferenceUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
