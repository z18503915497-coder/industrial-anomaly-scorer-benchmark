# ============================================================
# 工具模块: 随机种子 / 日志 / GPU管理 / IO
# ============================================================

import os
import random
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch


# ---- 随机种子 ----

def set_seed(seed: int = 42) -> None:
    """全局固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---- 日志 ----

def setup_logger(
    name: str = "experiment",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """初始化日志器：同时输出到文件和控制台。

    Args:
        name: 日志器名称
        log_dir: 日志输出目录
        level: 日志级别

    Returns:
        Logger 实例
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = Path(log_dir) / f"{timestamp}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 文件 handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---- GPU ----

def get_available_device() -> str:
    """自动检测可用设备。"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_device_info(logger: Optional[logging.Logger] = None) -> None:
    """打印当前设备信息。"""
    device = get_available_device()
    msg = f"设备: {device}"
    if device.startswith("cuda"):
        msg += f" | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    elif device == "cpu":
        import multiprocessing

        msg += f" | CPU cores: {multiprocessing.cpu_count()}"

    if logger:
        logger.info(msg)
    else:
        print(msg)


# ---- 结果保存 ----

def save_summary(summary: dict, output_path: str) -> None:
    """保存实验运行摘要为 JSON。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_csv(data: list, columns: list, output_path: str) -> None:
    """保存数据为 CSV。"""
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in data:
            if isinstance(row, dict):
                writer.writerow([row.get(c, "") for c in columns])
            else:
                writer.writerow(row)


class Timer:
    """简单的计时器上下文管理器。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    def __str__(self):
        return f"{self.name}: {self.elapsed:.2f}s"
