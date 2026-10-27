# ============================================================
# 配置管理模块
# 功能: 读取 YAML 配置 + 命令行覆盖 + 自动检测设备
# ============================================================

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


# ---- 项目根目录（从此文件向上两级） ----
PROJECT_ROOT = Path(__file__).parent.parent


def get_device() -> str:
    """自动检测可用设备: CUDA > MPS (macOS) > CPU"""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件，支持命令行覆盖。

    Args:
        config_path: YAML 配置文件路径 (相对于项目根目录或绝对路径)
        cli_overrides: 命令行覆盖字典, 如 {"model.name": "ViT-B/16"}

    Returns:
        配置字典
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行覆盖
    if cli_overrides:
        for key_path, value in cli_overrides.items():
            _set_nested(config, key_path, value)

    # 自动检测设备 (如果配置中未指定)
    if "model" in config and "device" not in config.get("model", {}):
        config.setdefault("model", {})["device"] = get_device()

    # 替换相对路径为绝对路径
    _resolve_paths(config)

    return config


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """设置嵌套字典的值, e.g. key_path='model.name' → d['model']['name']=value"""
    keys = key_path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _resolve_paths(config: dict) -> None:
    """将配置中的相对路径替换为绝对路径（基于项目根目录）"""
    path_keys = ["dataset.root", "output.root"]
    for key_path in path_keys:
        try:
            keys = key_path.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            val = d[keys[-1]]
            if not Path(val).is_absolute():
                d[keys[-1]] = str(PROJECT_ROOT / val)
        except (KeyError, TypeError):
            pass


def save_config_snapshot(config: dict, output_path: str) -> None:
    """保存配置快照到实验结果目录"""
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def build_cli_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="SCI_Q1_AI 实验系统")
    parser.add_argument("--config", type=str, default="config/pilot.yaml", help="配置文件路径")
    parser.add_argument("--model.name", type=str, dest="model_name", default=None)
    parser.add_argument("--model.device", type=str, dest="model_device", default=None)
    parser.add_argument("--seed", type=int, dest="seed", default=None)
    return parser


def parse_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """从 argparse 结果提取覆盖字典"""
    overrides = {}
    if args.model_name:
        overrides["model.name"] = args.model_name
    if args.model_device:
        overrides["model.device"] = args.model_device
    if args.seed is not None:
        overrides["experiment.seed"] = args.seed
    return overrides


def get_config_dir() -> Path:
    return PROJECT_ROOT / "config"


def get_data_dir() -> Path:
    return PROJECT_ROOT / "data"


def get_results_dir() -> Path:
    return PROJECT_ROOT / "results"

# ============================================================
# Experiment utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logger(name: str = "experiment"):
    """创建简单实验日志器."""
    import logging
    import sys

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def print_device_info(logger=None):
    """打印设备信息."""
    import torch

    msg = (
        f"CUDA available: {torch.cuda.is_available()}, "
        f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    if logger:
        logger.info(msg)
    else:
        print(msg)