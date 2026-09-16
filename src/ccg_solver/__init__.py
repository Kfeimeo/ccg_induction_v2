"""CCG 词典求解器：把词型的 CCG 范畴当作未知数，从无标注句子集合中解出来。

模块划分（按里程碑逐步填充）：
- ``category``  M0  范畴的 hash-cons 表示、解析/打印、复杂度度量
- ``state``     M0  全局状态：带 trail 的并查集 + 一阶合一 + occurs check
"""
