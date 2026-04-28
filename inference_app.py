#!/usr/bin/env python3
"""
推理应用启动文件
运行此文件启动PyQt5推理UI
"""
from ui.inference_ui import main
from src.utils.result_manager import ResultManager
from mysql_config import MYSQL_CONFIG

if __name__ == '__main__':
    result_manager = ResultManager(db_type="mysql", mysql_config=MYSQL_CONFIG)
    main()