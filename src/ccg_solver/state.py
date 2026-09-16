"""M0：全局状态（"增广矩阵"）——带 trail 的并查集 + 一阶合一（接口草案，待测试用例确认后实现）。

设计约定（与 tests/test_m0_unify.py、tests/test_m0_trail.py 对应）：

- ``unify(a, b) -> bool``：成功返回 True 并把必要的合并记入 trail；失败返回 False，
  且**原子性**：失败时状态回滚到调用前（trail 长度不变，途中做过的部分绑定全部撤销）。
- ``resolve(c)``：深度替换所有已绑定变量，返回 hash-cons 后的范畴；未绑定变量替换为其代表元。
- ``same(a, b)``：``resolve(a) is resolve(b)``。
- 合同闭包：前向（``X = Y`` ⇒ ``S/X = S/Y``）由 hash-cons + resolve 自动成立；
  反向（两复合项相等 ⇒ 子项对应合并）由 unify 递归完成，包括两个变量各自绑定到复合项后再合并的情况。
- occurs check：变量不得（直接或经绑定链）出现在将要绑定给它的项中。
- ``mark() -> int`` 返回当前 trail 位置；``undo_to(mark)`` 逐条撤销到该位置。向前撤销（mark 大于当前）抛 ``ValueError``。
- ``bindings() -> dict[Var, Category]``：dump 接口，列出所有非代表元变量到其 ``resolve`` 结果的映射。
"""
from __future__ import annotations

from .category import Category, Var


class State:
    def __init__(self) -> None:
        raise NotImplementedError

    def mark(self) -> int:
        raise NotImplementedError

    def undo_to(self, mark: int) -> None:
        raise NotImplementedError

    def unify(self, a: Category, b: Category) -> bool:
        raise NotImplementedError

    def resolve(self, c: Category) -> Category:
        raise NotImplementedError

    def same(self, a: Category, b: Category) -> bool:
        raise NotImplementedError

    def bindings(self) -> dict[Var, Category]:
        raise NotImplementedError
