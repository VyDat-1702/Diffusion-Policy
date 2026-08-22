"""Min-max linear normalizer mapping data to [-1, 1] per feature dimension."""

import numpy as np


class LinearNormalizer:
    """Per-dimension min-max normalizer with round-trip (de)normalization.

    Degenerate dimensions (min == max) are mapped with scale=1, offset=0
    so that normalization never produces NaN.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.offset_ = None

    def fit(self, data):
        """Compute per-dimension min/max from data along axis 0 and store scale/offset."""
        data = np.asarray(data, dtype=np.float64)
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        span = self.max_ - self.min_
        # Degenerate dims (min == max): scale=1, offset=0 to avoid NaN.
        self.scale_ = np.where(span > 0, span, 1.0)
        self.offset_ = np.where(span > 0, self.min_, 0.0)
        return self

    def normalize(self, x):
        """Map raw values to [-1, 1]: 2 * (x - min) / (max - min) - 1."""
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (x - self.offset_) / self.scale_ - 1.0

    def unnormalize(self, x):
        """Map normalized values back to raw: (x + 1) / 2 * (max - min) + min."""
        x = np.asarray(x, dtype=np.float64)
        return (x + 1.0) / 2.0 * self.scale_ + self.offset_

    def save(self, path):
        """Persist normalizer parameters to a .npz file."""
        np.savez(path, min_=self.min_, max_=self.max_,
                 scale_=self.scale_, offset_=self.offset_)

    def load(self, path):
        """Restore normalizer parameters from a .npz file."""
        data = np.load(path)
        self.min_ = data['min_']
        self.max_ = data['max_']
        self.scale_ = data['scale_']
        self.offset_ = data['offset_']
        return self