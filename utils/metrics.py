import numpy as np

def HR(ranked_list, ground_truth):
    hit = 0
    for item in ground_truth:
        if item in ranked_list:
            hit += 1
    return hit / len(ground_truth)


def NDCG(ranked_list, ground_truth):
    idcg = IDCG(len(ground_truth))
    dcg = DCG(ranked_list, ground_truth)
    return dcg / idcg


def DCG(ranked_list, ground_truth):
    dcg = 0
    for item in ground_truth:
        if item in ranked_list:
            rank = ranked_list.index(item)
            dcg += 1 / np.log2(rank + 2)
    return dcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / np.log2(i + 2)
    return idcg
