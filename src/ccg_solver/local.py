"""M1：局部求解器 —— 一句话 = 一行方程（接口草案）。

``solve_sentence(sent, lex, st)``：
- 在 ``st`` 之上枚举句子的所有推导（CKY，格子里是含变量的范畴），每条推导 = 一串合一；
  只保留与当前状态一致的（存活）。
- 返回前把状态回滚到调用前：**调用不改变 st**。
- 按 σ 去重：σ = 推导蕴含的词型变量指派（``trail[mark:]`` 限制到词型变量），局部新变量按首次出现
  顺序规范重命名为 ``?_1 ?_2 ...``（打印形式）；全局已有变量保持原样。σ 相同的推导折叠为一条，
  ``n_trees`` 记录折叠了多少棵树。
- 返回列表按 ``key`` 排序，因此输出确定。
- 剪枝：每次规则应用后，句中所有词型变量 resolve 后必须 ``is_admissible``。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .category import Category
from .corpus import Lexicon, Sentence
from .state import State


@dataclass(frozen=True)
class Tree:
    """推导树：叶子是词位置，内部节点是 (规则名, 左, 右)。"""
    rule: str | None
    span: tuple[int, int]
    children: tuple["Tree", ...] = ()

    def bracketed(self, words: tuple[str, ...]) -> str:
        """如 ``[< [> the cat] sleeps]``。"""
        raise NotImplementedError


@dataclass
class Derivation:
    key: tuple[str, ...]  # 句中每个 token 位置的 σ（规范打印），去重键
    sigma: dict[str, Category]  # 词型 → 范畴（含局部变量，未规范化；仅供 apply / dump）
    tree: Tree  # 代表树（折叠组里 bracketed 最小者）
    n_trees: int = 1

    def apply(self, st: State) -> bool:
        """把这条推导的词型指派并入全局状态（失败时 st 原子回滚）。"""
        raise NotImplementedError


def solve_sentence(sent: Sentence, lex: Lexicon, st: State) -> list[Derivation]:
    raise NotImplementedError


def enumerate_trees(n: int) -> int:
    """长度 n 的句子，规则集下的原始推导树数上界（Catalan(n-1) × |RULES|^(n-1)），供测试与 dump 用。"""
    raise NotImplementedError
