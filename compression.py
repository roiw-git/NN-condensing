from itertools import product
from time import time
from typing import Iterable

import numpy as np

from _constants import CompressionSchemes
from algo_utils import KNN, query_k_nn
from params.user_params.user_params_handler import CompressionParams
from scipy.spatial import Delaunay as delaunay


def get_compressed_proto_indices_based_on_neighbours_sets(num_of_prototypes: int,
                                                          neighbours_sets: Iterable[Iterable[int]],
                                                          prototypes_labels: Iterable) -> np.ndarray:
    consistent_prototypes_indices = np.ones(num_of_prototypes, dtype=bool)
    for inter_neighbours_indices_set in neighbours_sets:
        if np.all(np.logical_not(consistent_prototypes_indices[
                      inter_neighbours_indices_set])):  # if all the inter-neighbours are
            # already labeled-inconsistent -- there's no reason to check
            continue
        if np.unique(prototypes_labels[inter_neighbours_indices_set]).size > 1:  # means that the
            # neighbours are not labeled-consistent
            consistent_prototypes_indices[inter_neighbours_indices_set] = False
    return np.where(np.logical_not(consistent_prototypes_indices))[0]


def get_protoapprox_compressed_indices(prototypes_samples: np.ndarray, prototypes_labels: np.ndarray,
                                       train_samples: np.ndarray) -> np.ndarray:
    if np.unique(prototypes_labels).size <= 1:
        return np.arange(len(prototypes_samples))
    # two_first_neighbours_tree = KNN(prototypes_samples)
    # neighbours_sets = two_first_neighbours_tree.query(train_samples, k=2)
    neighbours_sets = query_k_nn(prototypes_samples, train_samples, k=2)
    num_of_prototypes = len(prototypes_samples)
    return get_compressed_proto_indices_based_on_neighbours_sets(num_of_prototypes, neighbours_sets, prototypes_labels)


def get_protocomp_compressed_indices(prototypes_samples: np.ndarray, prototypes_labels: np.ndarray,
                                     compression_params: CompressionParams) -> np.ndarray:
    if np.unique(prototypes_labels).size <= 1:
        return np.arange(len(prototypes_samples))
    num_of_samples = len(prototypes_samples)
    dim = compression_params.dim
    if num_of_samples <= dim + 1:
        return np.arange(num_of_samples)
    triangulation = delaunay(prototypes_samples)
    neighbours_sets = triangulation.simplices
    num_of_prototypes = len(prototypes_samples)
    return get_compressed_proto_indices_based_on_neighbours_sets(num_of_prototypes, neighbours_sets, prototypes_labels)


def get_greedy_weighted_heuristic_compressed_indices(prototypes_samples: np.ndarray, prototypes_labels: np.ndarray,
                                                     compression_params: CompressionParams) -> np.ndarray:
    num_of_prototypes = len(prototypes_samples)
    if num_of_prototypes <= 4:
        return np.arange(num_of_prototypes)
    distances = np.linalg.norm([[sample_b - sample_a for sample_b in prototypes_samples] for sample_a in prototypes_samples], axis=2)
    d_ne = np.empty(num_of_prototypes)
    for label in np.unique(prototypes_labels):
        labels_indices = prototypes_labels == label
        try:
            d_ne[labels_indices] = distances[labels_indices][:, np.logical_not(labels_indices)].min(axis=1)
        except Exception as e:
            print(e)
            # breakpoint()
            raise e
    s_prime = np.ones(num_of_prototypes, dtype=bool)
    compressed_indices = np.zeros(num_of_prototypes, dtype=bool)
    for _ in range(num_of_prototypes):
        # we don't use "while" to avoid the risk of an infinite loop
        if np.all(np.logical_not(s_prime)):
            break
        p_idx = np.argmax(np.sum(np.transpose(distances[:, s_prime].transpose() <= d_ne), axis=1))
        s_prime[distances[p_idx] <= d_ne[p_idx]] = False
        compressed_indices[p_idx] = True
    else:
        raise Exception(f'encountered an infinite loop')
    return np.where(compressed_indices)[0]


def get_compressed_indices(prototypes_samples: np.ndarray, prototypes_algo_labels: np.ndarray,
                           train_samples: np.ndarray,
                           compression_params: CompressionParams) -> [np.ndarray, float]:
    compression_scheme = compression_params.compression_scheme
    if compression_scheme == CompressionSchemes.NONE:
        compressed_indices = np.arange(len(prototypes_samples))
        compression_time = 0.0
        return compressed_indices, compression_time
    start_time = time()
    if compression_scheme == CompressionSchemes.PROTOAPPROX:
        compressed_indices = get_protoapprox_compressed_indices(prototypes_samples, prototypes_algo_labels,
                                                                train_samples)
    elif compression_scheme == CompressionSchemes.PROTOCOMP:
        compressed_indices = get_protocomp_compressed_indices(prototypes_samples, prototypes_algo_labels,
                                                              compression_params)
    else:
        raise Exception(f'compression scheme{compression_scheme} is not supported')
    compression_time = time() - start_time
    return compressed_indices, compression_time
