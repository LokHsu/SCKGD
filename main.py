import os
import random
from datetime import datetime

import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.lightgcn import LightGCN
from models.club import CLUBSample
from kg import KG, kg_pretrain
from utils import graph
from utils.dataloader import *
from utils.loss import *
from utils.evaluator import *
from utils.wandb_logger import WandbLogger
from params import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, gcn, club, gcn_opt, club_opt):
    print(f"{datetime.now()}: Negative Sampling start...")
    train_loader.dataset.neg_sample()
    print(f"{datetime.now()}: Negative Sampling ended.")

    epoch_loss = 0
    for u, i, j in tqdm(train_loader):
        user_embs, item_embs = gcn()

        bpr_loss, reg_loss = calc_bpr_loss(u, i, j, user_embs, item_embs, behaviors, device)
        reg_loss = args.l2_reg * reg_loss
        
        batch_users = u.long().to(device)

        infonce_loss = calc_infonce_loss(user_embs, batch_users, behaviors)
        infonce_loss = args.ssl_reg * infonce_loss

        item_mi = club(user_embs, batch_users)
        mi = args.club_reg * item_mi

        loss = (bpr_loss + reg_loss + infonce_loss + mi) / args.batch
        epoch_loss += loss.item()

        gcn_opt.zero_grad()
        loss.backward()
        gcn_opt.step()

    user_embs, item_embs = gcn()
    user_embs = [user_emb.detach() for user_emb in user_embs]
    item_embs = [item_emb.detach() for item_emb in item_embs]

    for _ in range(args.club_train_step):
        learning_loss = club.learning_loss(user_embs)

        club_opt.zero_grad()
        learning_loss.backward()
        club_opt.step()
    
    return epoch_loss


if __name__ == '__main__':
    set_random_seed()

    if args.dataset == 'Tmall':
        behaviors = ['pv', 'fav', 'cart', 'buy']

    elif args.dataset == 'IJCAI_15':
        behaviors = ['click', 'fav', 'cart', 'buy']

    elif args.dataset == 'retailrocket':
        behaviors = ['fav', 'cart', 'buy']

    elif args.dataset == 'Taobao':
        behaviors = ['pv', 'cart', 'buy']

    train_file = args.data_dir + args.dataset + '/trn_'
    test_file = args.data_dir + args.dataset + '/tst_int'

    train_u2i = []
    for i in range(len(behaviors)):
        with open(train_file + behaviors[i], 'rb') as f:
            u2i = pickle.load(f)
            train_u2i.append(u2i)

            if behaviors[i] == args.target:
                user_num = u2i.get_shape()[0]
                item_num = u2i.get_shape()[1]

    train_dataset = TrainDataset(train_u2i, behaviors, item_num)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch, num_workers=8)

    kg_dataset = KGDataset(args.data_dir + args.dataset + '/kg.npy')

    with open(test_file, 'rb') as f:
        test_dataset = TestDataset(pickle.load(f))
    test_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=4, pin_memory=True)

    adj_matrix = graph.create_adj_mats(train_u2i, user_num, item_num, behaviors, device)
    print(f"{user_num = }\n{item_num = }")

    head_embs = []
    relation_embs = []
    tail_embs = []
    for i, model in enumerate(['tp-transr', 'tp-tatec']):
        kg_file = os.path.join(args.data_dir, args.dataset, f'{model}.pth')
        if not os.path.exists(kg_file):
            kg_pretrain(item_num, behaviors, model)
        pretrain = torch.load(kg_file)
        kg = KG(item_num, behaviors, kg_dataset).to(device)
        kg.load_state_dict(pretrain)

        kg.eval()
        with torch.no_grad():
            head_emb, relation_emb, tail_emb, _ = kg()
        head_embs.append(head_emb.detach())
        relation_embs.append(relation_emb.detach())
        tail_embs.append(tail_emb.detach())

    gcn = LightGCN(user_num, item_num, behaviors, adj_matrix).to(device)
    club = CLUBSample(behaviors, kg_dataset.heads, kg_dataset.relations, kg_dataset.tails, head_embs, relation_embs, tail_embs).to(device)
    gcn_opt = torch.optim.Adam(gcn.parameters(), lr=args.lr)
    club_opt = torch.optim.Adam(club.parameters(), lr=args.lr)

    if args.wandb:
        wandb_logger = WandbLogger()

    for epoch in range(1, args.epochs + 1):
        print(f"\nTrainning Epoch {epoch}:")
        loss = train(train_loader, gcn, club, gcn_opt, club_opt)
        print(f"Epoch {epoch} Evaluation Metrics:\n{loss = :.6f}")
        with torch.no_grad():
            user_embs, item_embs = gcn()
            test_res = test(
                test_loader,
                train_u2i[-1],
                user_embs[-1].detach(),
                item_embs[-1].detach(),
            )
        if args.wandb:
            wandb_logger.log_metrics(epoch, loss, test_res, gcn)
