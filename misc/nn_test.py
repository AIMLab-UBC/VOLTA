import unittest

import torch

from misc.nn import NearestNeighbor


class TestNearestNeighbor(unittest.TestCase):

    def test_validity(self):
        def create_features(mu, size=100000, feature_dim=1000):
            features = []
            labels = []
            for i, m in enumerate(mu):
                features.append(torch.empty(size, feature_dim).normal_(mean=m, std=1))
                labels.extend([i for _ in range(size)])
            features = torch.cat(features, dim=0)
            labels = torch.LongTensor(labels)
            return features, labels

        test_cases = [
            ({'mu': [1, 10], 'size': 100, 'feature_dim': 10}, 1, 1),
            ({'mu': [1, 2], 'size': 100, 'feature_dim': 10}, 0.8, 1),
            ({'mu': [1, 1], 'size': 100, 'feature_dim': 10}, 0.4, 0.6),
        ]

        for case, min_range, max_range in test_cases:
            train_features, train_labels = create_features(**case)
            test_features, test_labels = create_features(**case)

            nn = NearestNeighbor()
            nn.fit(train_features, train_labels)
            prediction = nn.predict(test_features)
            acc = ((prediction == test_labels).sum() / len(prediction)).item()
            assert min_range <= acc <= max_range
