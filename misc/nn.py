import numpy as np
import torch


class NearestNeighbor:

    def __init__(self):
        self.train_features = None
        self.train_labels = None

    def fit(self, features: torch.Tensor or np.ndarray, labels: torch.Tensor or np.ndarray):
        self.train_features = features.clone()
        self.train_labels = labels.clone()
        return self

    def predict(self, features: torch.Tensor or np.ndarray):
        distance = ((features[:, None, :] - self.train_features[None, :, :]) ** 2).sum(-1)
        ind_nn = distance.argmin(dim=1)
        lab_nn = self.train_labels[ind_nn]
        return lab_nn.type(torch.LongTensor).to(features.device)
