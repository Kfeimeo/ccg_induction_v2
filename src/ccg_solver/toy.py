"""M2 玩具语料：20 条金标准词典，30 句，长度 ≤ 8，只用 > <。

满足 docs/m1-conclusion-m2-plan.md §5.3 与增补 3 的要求：
- VP 修饰语：quickly / often
- 最小对：the cat sleeps / the dog sleeps / the cat runs
- 同一动词出现在结构不同的宾语上下文：sees the dog / sees dogs / sees the big dog
"""
M2_GOLD = {
    "the": "NP/N", "a": "NP/N",
    "cat": "N", "dog": "N", "bird": "N", "fish": "N",
    "big": "N/N", "small": "N/N",
    "cats": "NP", "dogs": "NP", "birds": "NP", "john": "NP", "mary": "NP",
    "sleeps": "S[dcl]\\NP", "runs": "S[dcl]\\NP",
    "sees": "(S[dcl]\\NP)/NP", "chases": "(S[dcl]\\NP)/NP", "likes": "(S[dcl]\\NP)/NP",
    "quickly": "(S[dcl]\\NP)\\(S[dcl]\\NP)", "often": "(S[dcl]\\NP)\\(S[dcl]\\NP)",
}
M2_CORPUS = [
    "the cat sleeps .",
    "the dog sleeps .",
    "the cat runs .",
    "john sleeps .",
    "mary runs .",
    "the cat sees the dog .",
    "the dog sees the cat .",
    "the cat sees dogs .",
    "the cat sees the big dog .",
    "john sees mary .",
    "mary likes john .",
    "the dog chases the cat .",
    "the dog chases birds .",
    "the big dog chases the small cat .",
    "john likes the small bird .",
    "the bird sees the fish .",
    "the cat sees the dog quickly .",
    "the dog runs quickly .",
    "john sleeps often .",
    "mary sees birds often .",
    "the small cat sleeps .",
    "a dog sleeps .",
    "a cat sees a bird .",
    "the big cat runs quickly .",
    "john chases dogs .",
    "mary chases the big dog quickly .",
    "the fish sleeps .",
    "a small dog chases cats .",
    "john likes cats .",
    "the bird runs .",
]
