"""M2：域（接口草案）。复杂度界 L 下的全体 ground 候选范畴。"""
from __future__ import annotations

from .category import Category

# 域枚举用的原子。不含裸 S：金标准里 S 总带特征，裸 S 只会让域翻倍。
DEFAULT_ATOMS: tuple[tuple[str, str | None], ...] = (
    ("S", "dcl"), ("S", "q"), ("N", None), ("NP", None), ("PP", None), ("conj", None),
)


def enumerate_categories(L: int, atoms=DEFAULT_ATOMS) -> list[Category]:
    """所有 ground、``is_admissible``、``complexity ≤ L`` 的范畴。顺序确定（按 (complexity, str)）。"""
    raise NotImplementedError
