import numpy as np
import torch
import pandas as pd

from utils import metrics
from params import args

'''
One-to-One Negative Sampling
'''
def test(test_loader, tar_u2i, user_emb, item_emb):
    res = {'hr': 0, 'ndcg': 0}
    for user, item in test_loader:
        sampled_user, sampled_item = sample_uninter_items(tar_u2i, user, item)
        sampled_user_emb = user_emb[sampled_user]
        sampled_item_emb = item_emb[sampled_item]
    
        predict_score = torch.sum(torch.mul(sampled_user_emb, sampled_item_emb), dim=1)
        
        predict_score = predict_score.view(user.shape[0], 100)
        sampled_item = sampled_item.reshape((user.shape[0], 100))
        for i in range(user.shape[0]):
            _, topk_indices = torch.topk(predict_score[i], args.topks)
            topk_indices = topk_indices.cpu()
            ranked_list = sampled_item[i][topk_indices].tolist()

            ground_truth = [item[i].item()]
            HR = metrics.HR(ranked_list, ground_truth)
            NDCG = metrics.NDCG(ranked_list, ground_truth)
            res['hr'] += HR
            res['ndcg'] += NDCG

    res['hr'] = res['hr'] / test_loader.dataset.__len__()
    res['ndcg'] = res['ndcg'] / test_loader.dataset.__len__()

    for eval, value in res.items():
        print(f"{eval}@{args.topks} = {value:.6f}")

    return res


def sample_uninter_items(tar_u2i, batch_users, gt_items):
    sampled_user = np.repeat(batch_users.numpy(), 100)
    sampled_item = np.array([])

    tar_u2i = tar_u2i[batch_users].toarray()
    for i in range(len(batch_users)):
        negset = np.flatnonzero(tar_u2i[i] == 0)
        test_items = np.random.permutation(negset)[:99]
        test_items = np.append(test_items, gt_items[i])
        sampled_item = np.concatenate((sampled_item, test_items))

    return sampled_user, sampled_item
