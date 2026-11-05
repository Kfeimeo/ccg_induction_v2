"""玩具语料的一阶版金标准：同一 30 句（toy.M2_CORPUS），副词用 VP\\VP，动词用 VP、VP/NP。信息量与原金标准相同。"""
from .toy import M2_CORPUS  # noqa: F401

M2_GOLD_V2 = {
    "the": "NP/N", "a": "NP/N",
    "cat": "N", "dog": "N", "bird": "N", "fish": "N",
    "big": "N/N", "small": "N/N",
    "cats": "NP", "dogs": "NP", "birds": "NP", "john": "NP", "mary": "NP",
    "sleeps": "VP", "runs": "VP",
    "sees": "VP/NP", "chases": "VP/NP", "likes": "VP/NP",
    "quickly": "VP\\VP", "often": "VP\\VP",
}
