import torch
import numpy as np
import torch.nn.functional as F

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    return torch.sum(F.softplus(neg_score - pos_score))


def reg_loss(user_emb, pos_emb, neg_emb):
    reg_loss = 0.5 * (user_emb.norm(2).pow(2) +
                      pos_emb.norm(2).pow(2) +
                      neg_emb.norm(2).pow(2))
    return reg_loss


def infonce_loss(tar_user_emb, aux_user_emb, tau=0.5):
    x_norm = F.normalize(tar_user_emb)
    y_norm = F.normalize(aux_user_emb)
    
    pos_score = torch.sum(x_norm * y_norm, dim=1)
    pos_score = torch.exp(pos_score / tau)

    neg_score = torch.matmul(x_norm, y_norm.T)
    neg_score = torch.sum(torch.exp(neg_score / tau), dim=1)

    cl_loss = -torch.sum(torch.log(pos_score / neg_score))
    return cl_loss


def calc_bpr_loss(user, item_i, item_j, user_embs, item_embs, behaviors, device):
    bpr_loss_list = [None] * len(behaviors)
    for i in range(len(behaviors)):
        act_user_idx = np.where(item_i[i].cpu().numpy() != -1)[0]

        act_user = user[act_user_idx].long().to(device)
        act_user_pos = item_i[i][act_user_idx].long().to(device)
        act_user_neg = item_j[i][act_user_idx].long().to(device)

        act_user_emb = user_embs[i][act_user]
        act_user_pos_emb = item_embs[i][act_user_pos]
        act_user_neg_emb = item_embs[i][act_user_neg]

        bpr_loss_list[i] = bpr_loss(act_user_emb, act_user_pos_emb, act_user_neg_emb)
        
        if i == len(behaviors) - 1:
            l2_loss = reg_loss(act_user_emb, act_user_pos_emb, act_user_neg_emb)

    return sum(bpr_loss_list) / len(bpr_loss_list), l2_loss


def calc_infonce_loss(user_embs, batch_users, behaviors):
    cl_loss_list = [None] * len(behaviors)
    tar_user_emb = user_embs[-1][batch_users]
    for i in range(len(behaviors)):
        aux_user_emb = user_embs[i][batch_users]

        cl_loss_list[i] = infonce_loss(tar_user_emb, aux_user_emb)

    return sum(cl_loss_list) / len(cl_loss_list)
