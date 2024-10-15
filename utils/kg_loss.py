import torch
import torch.nn.functional as F
from params import args


def kg_reg_loss(*embeddings):
    reg_loss = 0.5 * sum(emb.norm(2).pow(2)
                         for emb in embeddings) / embeddings[0].shape[0]
    return args.l2_reg * reg_loss


def TP_TransR(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat):
        pos_h = torch.matmul(r_mat, pos_h)
        neg_h = torch.matmul(r_mat, neg_h)

        pos_t = torch.matmul(r_mat, pos_t)
        neg_t = torch.matmul(r_mat, neg_t)

        pos_score = torch.sum(torch.pow(pos_h + pos_r - pos_t, 2),
                              dim=1)
        neg_score = torch.sum(torch.pow(neg_h + neg_r - neg_t, 2),
                              dim=1)

        kg_loss = -torch.mean(F.logsigmoid(neg_score - pos_score))
        l2_loss = kg_reg_loss(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat)
        return kg_loss + l2_loss


def TP_TATEC(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat):
    pos_mrt = torch.matmul(r_mat, pos_t)
    neg_mrt = torch.matmul(r_mat, neg_t)
    
    pos_hmrt = torch.sum(pos_h * pos_mrt, dim=1)
    neg_hmrt = torch.sum(neg_h * neg_mrt, dim=1)

    pos_hr = torch.sum(pos_h * pos_r, dim=1)
    neg_hr = torch.sum(neg_h * neg_r, dim=1)

    pos_tr = torch.sum(pos_t * pos_r, dim=1)
    neg_tr = torch.sum(neg_t * neg_r, dim=1)

    pos_ht = torch.sum(pos_h * pos_t, dim=1)
    neg_ht = torch.sum(neg_h * neg_t, dim=1)

    pos_score = pos_hmrt + pos_hr + pos_tr + pos_ht
    neg_score = neg_hmrt + neg_hr + neg_tr + neg_ht

    kg_loss = -torch.mean(F.logsigmoid(neg_score - pos_score))
    l2_loss = kg_reg_loss(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat)
    return kg_loss + l2_loss


def TransR(h_emb, r_emb, pos_t_emb, neg_t_emb, r_mat):
        h_emb = h_emb.unsqueeze(-1)
        r_emb = r_emb.unsqueeze(-1)
        pos_t_emb = pos_t_emb.unsqueeze(-1)
        neg_t_emb = neg_t_emb.unsqueeze(-1)

        h_emb = torch.matmul(r_mat, h_emb)
        pos_t_emb = torch.matmul(r_mat, pos_t_emb)
        neg_t_emb = torch.matmul(r_mat, neg_t_emb)

        pos_score = torch.sum(torch.pow(h_emb + r_emb - pos_t_emb, 2),
                              dim=1)
        neg_score = torch.sum(torch.pow(h_emb + r_emb - neg_t_emb, 2),
                              dim=1)

        kg_loss = -torch.mean(F.logsigmoid(neg_score - pos_score))
        return kg_loss + kg_reg_loss(h_emb, r_emb, pos_t_emb, neg_t_emb, r_mat)


def TATEC(h_emb, r_emb, pos_t_emb, neg_t_emb, r_mat):
    h_emb = h_emb.unsqueeze(-1)
    r_emb = r_emb.unsqueeze(-1)
    pos_t_emb = pos_t_emb.unsqueeze(-1)
    neg_t_emb = neg_t_emb.unsqueeze(-1)

    pos_mrt = torch.matmul(r_mat, pos_t_emb)
    neg_mrt = torch.matmul(r_mat, neg_t_emb)
    
    pos_hmrt = torch.sum(h_emb * pos_mrt, dim=1)
    neg_hmrt = torch.sum(h_emb * neg_mrt, dim=1)

    hr = torch.sum(h_emb * r_emb, dim=1)

    pos_tr = torch.sum(pos_t_emb * r_emb, dim=1)
    neg_tr = torch.sum(neg_t_emb * r_emb, dim=1)

    pos_ht = torch.sum(h_emb * pos_t_emb, dim=1)
    neg_ht = torch.sum(h_emb * neg_t_emb, dim=1)

    pos_score = pos_hmrt + hr + pos_tr + pos_ht
    neg_score = neg_hmrt + hr + neg_tr + neg_ht

    kg_loss = -torch.mean(F.logsigmoid(neg_score - pos_score))
    return kg_loss + kg_reg_loss(h_emb, r_emb, pos_t_emb, neg_t_emb, r_mat)
