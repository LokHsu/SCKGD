import torch
import torch.nn as nn

from params import args

class CLUBSample(nn.Module):
    def __init__(self, behaviors, heads, relations, tails, head_embs, relation_embs, tail_embs):
        super(CLUBSample, self).__init__()
        self.behaviors = behaviors

        x_dim = args.emb_size
        y_dim = args.emb_size
        hidden_size = 2 * args.emb_size

        self.heads = torch.tensor(heads).cuda()
        self.relations = torch.tensor(relations).cuda()
        self.tails = torch.tensor(tails).cuda()
        self.head_embs = head_embs
        self.relation_embs = relation_embs
        self.tail_embs = tail_embs

        self.fc = nn.Sequential(nn.Linear(x_dim, y_dim),
                                nn.ReLU())
        
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, y_dim),
                                      nn.Tanh())


    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    

    def forward(self, item_emb, batch_items):
        mi_list = [None] * len(self.head_embs)
        for i in range(len(self.head_embs)):
            batch_kg_items = batch_items[torch.isin(batch_items, self.heads)]

            tar_item_emb = item_emb[-1][batch_kg_items]
            head_emb = self.head_embs[i][batch_kg_items]
            mi_list[i] = self.calc_mi_est(tar_item_emb, head_emb)
        
            for beh in range(len(self.behaviors) - 1):
                tails = self.tails[torch.nonzero(self.relations == beh).squeeze()]
                batch_t = batch_items[torch.isin(batch_items, tails)]
                
                aux_item_emb = item_emb[beh][batch_t]

                tail_emb = self.tail_embs[i][batch_t]
                relation_emb = self.relation_embs[i][beh].expand(tail_emb.shape)
                agg_emb = self.fc(tail_emb + relation_emb)

                mi_list[i] += self.calc_mi_est(aux_item_emb, agg_emb)
        return sum(mi_list) / len(mi_list)
    

    def calc_mi_est(self, x_samples, y_samples):
        if torch.numel(x_samples) == 0 or torch.numel(y_samples) == 0:
            return torch.tensor(0.0, device=x_samples.device)
        
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        negative = (-(mu - y_samples[random_index]) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        mi = (positive - negative).sum()

        return torch.clamp(mi / 2., min=0.0)


    def learning_loss(self, item_embs):
        loss_list = [None] * len(self.head_embs)
        for i in range(len(self.head_embs)):
            kg_items = self.heads.t()

            tar_item_emb = item_embs[-1][kg_items]
            head_emb = self.head_embs[i][kg_items]
            loss_list[i] = -self.loglikeli(tar_item_emb, head_emb)

            for beh in range(len(self.behaviors) - 1):
                t_items = self.tails.t()
                
                aux_item_emb = item_embs[beh][t_items]
                
                tail_emb = self.tail_embs[i][t_items]
                relation_emb = self.relation_embs[i][beh].expand(tail_emb.shape)
                agg_emb = self.fc(tail_emb + relation_emb)
                
                loss_list[i] += -self.loglikeli(aux_item_emb, agg_emb)
        return sum(loss_list) / len(loss_list)
        

    def loglikeli(self, x_samples, y_samples):
        if torch.numel(x_samples) == 0 or torch.numel(y_samples) == 0:
            return torch.tensor(0.0, device=x_samples.device)
        
        mu, logvar = self.get_mu_logvar(x_samples)
        llh = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1).mean()
        return llh
