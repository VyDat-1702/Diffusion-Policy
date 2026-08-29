import numpy as np


class LinearNormalizer:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.offset_ = None

    def fit(self, data, dim=None):

        data = np.asarray(data, dtype=np.float64)
        repeat = 1
        if dim is not None:
            repeat = data.shape[-1] // dim
            data = data.reshape(-1, dim)
        self.min_ = np.tile(data.min(axis=0), repeat)
        self.max_ = np.tile(data.max(axis=0), repeat)
        span = self.max_ - self.min_
        self.scale_ = np.where(span > 0, span, 1.0)
        self.offset_ = np.where(span > 0, self.min_, 0.0)
        return self

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (x - self.offset_) / self.scale_ - 1.0

    def unnormalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x + 1.0) / 2.0 * self.scale_ + self.offset_

    def save(self, path):
        np.savez(path, min_=self.min_, max_=self.max_,
                 scale_=self.scale_, offset_=self.offset_)

    def load(self, path):
        data = np.load(path)
        self.min_ = data['min_']
        self.max_ = data['max_']
        self.scale_ = data['scale_']
        self.offset_ = data['offset_']
        return self