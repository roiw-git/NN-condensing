from collections import Counter
from typing import Iterable, Union

import numpy as np
from sklearn.neighbors import KDTree


class KNN:
    def __init__(self, train_samples: np.ndarray):
        # self._train_samples = train_samples  # for debug purposes
        self.construction_tree = KDTree(train_samples)

    def query(self, query_samples: np.ndarray, k: int) -> np.ndarray:
        try:
            nn_indices = self.construction_tree.query(query_samples, k=k, return_distance=False)
        except Exception as e:
            raise e
        return nn_indices

def query_k_nn(train_samples: np.ndarray, query_samples: np.ndarray, k: int, weights: Union[None, np.ndarray, list] = None)->np.ndarray:
    if weights is None:
        construction_tree = KDTree(train_samples)
        nn_indices = construction_tree.query(query_samples, k=k, return_distance=False)
        return nn_indices
    # weights is not None
    if k != 1:
        raise NotImplemented(f'weighted k-NN is supported for k != 1')
    weights = np.reciprocal(weights)
    nn_indices = np.argmin(np.multiply(np.linalg.norm([[train_sample - query_sample for train_sample in train_samples] for query_sample in query_samples], axis=2), weights), axis=1)
    return nn_indices




def get_majority_votes(queries_candidates_indices: Iterable[Iterable[int]], candidates_labels: np.ndarray) -> np.ndarray:
    return np.array([Counter(candidates_labels[query_candidates_indices]).most_common(1)[0][0] for query_candidates_indices in queries_candidates_indices])

def compute_distances(point, set):
    return np.linalg.norm([point - p for p in set], axis =1)