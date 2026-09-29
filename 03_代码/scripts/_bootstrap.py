"""让脚本不论从哪个目录运行都能 import src。"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
DEFAULT_CFG = os.path.join(CODE_DIR, "configs", "default.yaml")
