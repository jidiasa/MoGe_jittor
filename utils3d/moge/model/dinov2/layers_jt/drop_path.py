# jt_drop_path.py

import jittor as jt
import jittor.nn as nn

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = jt.bernoulli(jt.full(shape, keep_prob))
    if keep_prob > 0.0:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())
