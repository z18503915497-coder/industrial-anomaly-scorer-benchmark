"""Leakage Checker — 验证 zero-shot 协议的严格性

检查项:
  1. target category 完全排除在 training set 外
  2. test split 未被用于 training
  3. defect masks 未被用于 SP 训练
"""
from pathlib import Path
from typing import List, Set


def check_target_exclusion(
    auxiliary_categories: List[str],
    target_category: str,
) -> dict:
    """检查 target category 是否完全排除。"""
    passed = target_category not in auxiliary_categories
    return {
        "check": "target_exclusion",
        "passed": passed,
        "detail": f"target='{target_category}' in aux={auxiliary_categories}" if not passed else "target excluded from auxiliary set",
    }


def check_test_exclusion(
    train_paths: List[Path],
    test_paths: List[Path],
) -> dict:
    """检查 training 数据是否混入 test 样本。"""
    train_stems = {p.stem for p in train_paths}
    test_stems = {p.stem for p in test_paths}
    overlap = train_stems & test_stems
    passed = len(overlap) == 0
    return {
        "check": "test_exclusion",
        "passed": passed,
        "detail": f"overlap={len(overlap)} files" if not passed else f"train={len(train_stems)} test={len(test_stems)} no overlap",
    }


def check_mask_usage(
    uses_defect_masks: bool,
) -> dict:
    """检查 defect masks 是否被用于 SP 训练。"""
    passed = not uses_defect_masks
    return {
        "check": "mask_usage",
        "passed": passed,
        "detail": "defect masks NOT used in training" if passed else "defect masks used — violates strict zero-shot protocol",
    }


def check_protocol(
    protocol: str,
    target_category: str,
    auxiliary_categories: List[str],
    train_paths: List[Path],
    test_paths: List[Path],
    uses_defect_masks: bool,
) -> dict:
    """综合泄漏检查。"""
    checks = [
        check_target_exclusion(auxiliary_categories, target_category),
        check_test_exclusion(train_paths, test_paths),
    ]

    if protocol == "strict_zero_shot":
        checks.append(check_mask_usage(uses_defect_masks))

    all_pass = all(c["passed"] for c in checks)

    return {
        "protocol": protocol,
        "target": target_category,
        "n_auxiliary": len(auxiliary_categories),
        "all_pass": all_pass,
        "checks": checks,
    }


def format_check_report(check_result: dict) -> str:
    """格式化泄漏检查报告。"""
    lines = [
        "=" * 40,
        "LEAKAGE CHECK",
        "=" * 40,
        f"Protocol: {check_result['protocol']}",
        f"Target:   {check_result['target']}",
        f"Auxiliary: {check_result['n_auxiliary']} categories",
        f"Overall:  {'PASS' if check_result['all_pass'] else 'FAIL'}",
    ]
    for ck in check_result["checks"]:
        status = "PASS" if ck["passed"] else "FAIL"
        lines.append(f"  [{status}] {ck['check']}: {ck['detail']}")

    return "\n".join(lines)
