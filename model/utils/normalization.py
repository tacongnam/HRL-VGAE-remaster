import numpy as np
import torch
from typing import Union


class Normalizer:

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.fitted = mean is not None and std is not None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1.0

        self.fitted = True
        return self

    def normalize(self, data):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before normalization")

        if isinstance(data, torch.Tensor):
            is_tensor = True
            data = data.cpu().numpy()
        else:
            is_tensor = False

        normalized = (data - self.mean) / self.std

        if is_tensor:
            normalized = torch.FloatTensor(normalized)

        return normalized

    def denormalize(self, data):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before denormalization")

        if isinstance(data, torch.Tensor):
            is_tensor = True
            data = data.cpu().numpy()
        else:
            is_tensor = False

        denormalized = data * self.std + self.mean

        if is_tensor:
            denormalized = torch.FloatTensor(denormalized)

        return denormalized

    def get_params(self):
        return {
            'mean': self.mean,
            'std': self.std,
        }

    def set_params(self, params):
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = True


class MinMaxScaler:

    def __init__(self, feature_min=None, feature_max=None):
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.fitted = feature_min is not None and feature_max is not None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.feature_min = np.min(data, axis=0)
        self.feature_max = np.max(data, axis=0)

        self.fitted = True
        return self

    def scale(self, data):
        if not self.fitted:
            raise RuntimeError("MinMaxScaler must be fitted before scaling")

        if isinstance(data, torch.Tensor):
            is_tensor = True
            data = data.cpu().numpy()
        else:
            is_tensor = False

        range_val = self.feature_max - self.feature_min
        range_val[range_val == 0] = 1.0

        scaled = (data - self.feature_min) / range_val

        if is_tensor:
            scaled = torch.FloatTensor(scaled)

        return scaled

    def inverse_scale(self, data):
        if not self.fitted:
            raise RuntimeError("MinMaxScaler must be fitted before inverse scaling")

        if isinstance(data, torch.Tensor):
            is_tensor = True
            data = data.cpu().numpy()
        else:
            is_tensor = False

        range_val = self.feature_max - self.feature_min
        range_val[range_val == 0] = 1.0

        unscaled = data * range_val + self.feature_min

        if is_tensor:
            unscaled = torch.FloatTensor(unscaled)

        return unscaled

    def get_params(self):
        return {
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
        }

    def set_params(self, params):
        self.feature_min = params['feature_min']
        self.feature_max = params['feature_max']
        self.fitted = True