import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnnの再現性確保オプション
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_event_label():
    fname = os.path.join(
                os.path.dirname(__file__),
                "data/event_label/output.csv"
    )
    ret = dict()
    for l in open(fname, "r").readlines():
        l = l.strip("\n")
        aid, eids = l.split(":")
        ret[int(aid)] = [int(x) for x in eids.split(",")]
    return ret

