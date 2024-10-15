import numpy as np
import torch
import pickle
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader

from utils.dataloader import KGDataset
from utils.loss import *
from utils.kg_loss import *
from params import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KG(nn.Module):
    def __init__(self, item_num, behaviors, kg_dataset):
        super(KG, self).__init__()
        self.item_num = item_num
        self.beh_num = len(behaviors)

        self.entity_num = kg_dataset.entity_num

        self.temp_size = args.temp_size
        self.emb_size = args.emb_size

        self._init_embeddings()


    def _init_embeddings(self,):
        self.head_emb = nn.Embedding(self.item_num, self.emb_size)
        self.relation_emb = nn.Embedding(self.beh_num, self.emb_size)
        self.tail_emb = nn.Embedding(self.entity_num, self.emb_size)
        self.r_mat = nn.Parameter(torch.Tensor(self.beh_num, self.emb_size, self.emb_size))
        self.w_t = nn.Parameter(torch.Tensor(self.temp_size, self.emb_size))

        init.normal_(self.head_emb.weight, std=0.1)
        init.normal_(self.relation_emb.weight, std=0.1)
        init.normal_(self.tail_emb.weight, std=0.1)
        init.xavier_uniform_(self.r_mat, gain=init.calculate_gain('relu'))
        init.xavier_uniform_(self.w_t, gain=init.calculate_gain('relu'))


    def project(self, emb, temp):
        w_t = F.normalize(self.w_t, p=2, dim=-1)[temp]
        
        proj_length = torch.sum(w_t * emb, dim=1, keepdim=True)
        proj_vector = proj_length.expand(-1, emb.shape[1]) * w_t
        proj = emb - proj_vector
        return proj


    def forward(self,):
        return self.head_emb.weight, \
               self.relation_emb.weight, \
               self.tail_emb.weight, \
               self.r_mat


def train(kg_loader, kg, kg_opt, model):
    epoch_loss = 0
    for h, r, pos_t, pos_temp, neg_t, neg_temp in kg_loader:
        head_emb, relation_emb, tail_emb, r_mat = kg()

        h_emb = head_emb[h.long().to(device)]
        r_emb = relation_emb[r.long().to(device)]
        pos_t_emb = tail_emb[pos_t.long().to(device)]
        neg_t_emb = tail_emb[neg_t.long().to(device)]
        r_mat = r_mat[r]
        
        pos_h = kg.project(h_emb, pos_temp).unsqueeze(-1)
        neg_h = kg.project(h_emb, neg_temp).unsqueeze(-1)
        pos_r = kg.project(r_emb, pos_temp).unsqueeze(-1)
        neg_r = kg.project(r_emb, neg_temp).unsqueeze(-1)
        pos_t = kg.project(pos_t_emb, pos_temp).unsqueeze(-1)
        neg_t = kg.project(neg_t_emb, neg_temp).unsqueeze(-1)

        if model == 'tp-transr':
            kg_loss = TP_TransR(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat) / args.batch
        elif model == 'tp-tatec':
            kg_loss = TP_TATEC(pos_h, neg_h, pos_r, neg_r, pos_t, neg_t, r_mat) / args.batch
        epoch_loss += kg_loss.item()
        
        kg_opt.zero_grad()
        kg_loss.backward()
        kg_opt.step()

    return epoch_loss


def kg_pretrain(item_num, behaviors, model='tp-transr'):
    kg_dataset = KGDataset(args.data_dir + args.dataset + '/kg.npy')
    kg_loader = DataLoader(kg_dataset, shuffle=True, batch_size=args.batch, num_workers=8)

    kg = KG(item_num, behaviors, kg_dataset).to(device)
    kg_opt = torch.optim.Adam(kg.parameters(), lr=args.lr)

    kg.train()
    best_loss = np.inf
    for epoch in range(1, 301):
        print(f"\nTrainning KG Epoch {epoch}:")
        loss = train(kg_loader, kg, kg_opt, model)
        print(f"{model} loss = {loss:.6f}")
        if loss < best_loss:
            print(f'Best {model} loss, save checkpoint..')
            best_loss = loss
            torch.save(kg.state_dict(), f'{args.data_dir}/{args.dataset}/{model}.pth')

    print(f"KG ({model}) Trainning Finish.")
    kg.eval()


if __name__ == "__main__":
    train_file = args.data_dir + args.dataset + '/trn_'

    if args.dataset == 'Tmall':
        behaviors = ['pv', 'fav', 'cart', 'buy']

    elif args.dataset == 'IJCAI_15':
        behaviors = ['click', 'fav', 'cart', 'buy']

    elif args.dataset == 'retailrocket':
        behaviors = ['fav', 'cart', 'buy']

    with open(train_file + behaviors[-1], 'rb') as f:
        u2i = pickle.load(f)
        item_num = u2i.get_shape()[1]

    kg_pretrain(item_num, behaviors, 'tp-transr')
    kg_pretrain(item_num, behaviors, 'tp-tatec')