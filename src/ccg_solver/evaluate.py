"""M3/§8：评测 —— 原子范畴重命名匹配 + 词型级准确率（接口草案）。"""
from __future__ import annotations


def match_atoms(solution: dict[str, str], gold: dict[str, str]) -> dict[str, str]:
    """二分图最大匹配：解出的原子/自由变量名 → 金标准原子名。匹配权 = 让多少词型的范畴完全一致。
    返回重命名表（解侧名 → 金标准名）。"""
    raise NotImplementedError


def accuracy(solution: dict[str, str], gold: dict[str, str]) -> tuple[float, dict[str, str]]:
    """(词型级准确率, 重命名后的解)。"""
    raise NotImplementedError
