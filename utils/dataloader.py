import random

import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, u2i):
        super(TestDataset, self).__init__()
        self.u2i = [[idx, i] for idx, i in enumerate(u2i) if i is not None]

    def __len__(self):
        return len(self.u2i)

    def __getitem__(self, idx):
        user = self.u2i[idx][0]
        item = self.u2i[idx][1]
        return user, item


class TrainDataset(Dataset):
    def __init__(self, u2i, behaviors, item_num):  
        super(TrainDataset, self).__init__()
        self.u2i = u2i

        self.item_num = item_num
        self.beh_num = len(behaviors)
        self.inter_users = u2i[-1].nonzero()[0]
        self.total_ints = self.inter_users.shape[0]

        self.pos_item = [[-1] * self.beh_num for _ in range(self.total_ints)]
        self.neg_item = [[-1] * self.beh_num for _ in range(self.total_ints)]

    def neg_sample(self):
        for beh in range(self.beh_num):
            u2i = self.u2i[beh]
            u2i_dok = u2i.todok()
            active_items = u2i.nonzero()[1]

            for i in range(self.total_ints):
                user = self.inter_users[i]

                if beh == (self.beh_num - 1):
                    self.pos_item[i][beh] = active_items[i]
                elif len(u2i[user].data) != 0:
                    iter_items = u2i[user].nonzero()[1]
                    self.pos_item[i][beh] = np.random.choice(iter_items)
                else:
                    self.pos_item[i][beh] = -1

                neg = np.random.randint(0, self.item_num)
                while (user, neg) in u2i_dok:
                    neg = np.random.randint(0, self.item_num)
                self.neg_item[i][beh] = neg

    def __len__(self):
        return self.total_ints

    def __getitem__(self, idx):
        user = self.inter_users[idx]
        pos = self.pos_item[idx]
        neg = self.neg_item[idx]
        return user, pos, neg


class KGDataset(Dataset):
    def __init__(self, kg_path):
        kg = np.load(kg_path)

        self.entity_num = kg[:, -2].max() + 1

        self.kg_h2t = defaultdict(list)
        for h, r, t, temp in kg:
            # key: h, value: (r, t, temporal)
            self.kg_h2t[h].append((r, t, temp))

        self.heads = list(self.kg_h2t.keys())

        self.relations = [r for value in self.kg_h2t.values() for r, _, _ in value]
        self.tails = [t for value in self.kg_h2t.values() for _, t, _ in value]


    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        h = self.heads[idx]
        r, pos_t, pos_temp = random.choice(self.kg_h2t[h])
        while True:
            neg_h = random.choice(self.heads)
            neg_t, neg_temp = random.choice(self.kg_h2t[neg_h])[1:]
            if (r, neg_t, neg_temp) not in self.kg_h2t[h]:
                break
        return h, r, pos_t, pos_temp, neg_t, neg_temp
