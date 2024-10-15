import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from params import args


"""
Pytorch Implementation of LightGCN from
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
by Xiangnan He et al.
"""
class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, behavior, norm_adj):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.beh_num = len(behavior)
        self.norm_adj = norm_adj

        self.emb_size = args.emb_size
        self.n_layers = 3

        self._init_embeddings()

        self.attention = Attention(self.emb_size, self.beh_num)

    def _init_embeddings(self,):
        self.user_beh_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_beh_emb = nn.Embedding(self.item_num, self.emb_size)
        init.xavier_uniform_(self.user_beh_emb.weight)
        init.xavier_uniform_(self.item_beh_emb.weight)


    def forward(self,):
        behavior_list = [None] * self.beh_num
        
        for i in range(self.beh_num):
            behavior_list[i] = self.propagate(
                self.norm_adj[i], self.user_beh_emb.weight, self.item_beh_emb.weight)
        
        bia_beh_list = self.attention(torch.stack(behavior_list, dim=1))
        user_beh_emb, item_beh_emb = torch.split(
            bia_beh_list, [self.user_num, self.item_num], dim=1)

        return user_beh_emb, item_beh_emb

    def propagate(self, adj, user_emb, item_emb):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(0, self.n_layers):
            if adj.is_sparse is True:
                ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            else:
                ego_embeddings = torch.mm(adj, ego_embeddings)
            
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        return all_embeddings


class Attention(nn.Module):
    def __init__(self, emb_size, beh_num):
        super(Attention, self).__init__()
        self.emb_size = emb_size
        self.beh_num = beh_num

        self._init_weights()

    def _init_weights(self,):
        self.trans_weights_s1 = [None] * self.beh_num
        self.trans_weights_s2 = [None] * self.beh_num
        for i in range(self.beh_num):
            self.trans_weights_s1[i] = nn.Parameter(torch.Tensor(self.emb_size, int(self.emb_size / 4))).cuda()
            self.trans_weights_s2[i] = nn.Parameter(torch.Tensor(int(self.emb_size / 4), 1)).cuda()
            init.xavier_uniform_(self.trans_weights_s1[i])
            init.xavier_uniform_(self.trans_weights_s2[i])

    def forward(self, behavior_lst):
        bias_beh_lst  = [None] * self.beh_num
        for i in range(self.beh_num):
            attention = F.softmax(
                torch.matmul(
                    torch.tanh(torch.matmul(behavior_lst, self.trans_weights_s1[i])),
                    self.trans_weights_s2[i]
                ).squeeze(2),
                dim=1
            ).unsqueeze(1)
            bias_beh_lst[i] = torch.matmul(attention, behavior_lst).squeeze(1)
        
        return torch.stack(bias_beh_lst, dim=0)
            
