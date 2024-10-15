import numpy as np
import scipy.sparse as sp
import torch

np.seterr(divide='ignore', invalid='ignore')

def create_adj_mats(train_u2i, user_num, item_num, behaviors, device):
    adj_matrix = [None] * len(behaviors)
    for i in range(len(behaviors)):
        adj_matrix[i] = create_adj_mat(train_u2i[i], user_num, item_num).to(device)
    return adj_matrix

def create_adj_mat(train_u2i, n_users, n_items):
    num_nodes = n_users + n_items

    train_u2i = (train_u2i != 0) * 1
    adj = sp.coo_matrix(train_u2i)

    users_np, items_np = adj.nonzero()
    ratings = adj.data
    
    tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+n_users)), shape=(num_nodes, num_nodes))
    adj_mat = tmp_adj + tmp_adj.T

    rowsum = np.array(adj_mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)
    return matrix_to_tensor(adj_matrix)

def matrix_to_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()
