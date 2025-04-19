import time
import numpy as np
import torch
import random
import os

def set_seed_tf(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time,
                                                  self._name,
                                                  self.run_time,
                                                  self.stage_time,
                                                  message))

    def update(self):
        self.current_time = time.time()
        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self

def bpr_neg_samp(uni_users, n_users, support_dict, item_array):
    users = np.random.choice(uni_users, size=n_users, replace=True)
    pos_items = []
    for user in users:
        pos_candidates = support_dict[user]
        pos_item = random.choice(pos_candidates)
        pos_items.append(pos_item)
    pos_items = np.array(pos_items, dtype=np.int64)
    neg_items = np.random.choice(item_array, len(users), replace=True)
    ret = torch.tensor(np.stack([users, pos_items, neg_items], axis=1), dtype=torch.long)
    return ret
