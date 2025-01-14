import cProfile
from copy import deepcopy
from itertools import combinations

import pprofile
from time import time
from typing import List, Iterable, Union
from warnings import warn

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from _constants import ClassificationSchemes
from algo_utils import KNN, get_majority_votes, query_k_nn, compute_distances
from params.user_params.user_params_handler import ClassificationParams


def get_proto_labels_of_proto_k_nn(prototypes_samples: np.ndarray, train_samples: np.ndarray, train_labels: np.ndarray,
                                   classification_params: ClassificationParams) -> np.ndarray:
    neighbours_indices = query_k_nn(train_samples, query_samples=prototypes_samples, k=classification_params.k)
    return get_majority_votes(neighbours_indices, train_labels)


def get_gamma_net_indices(prototypes_samples: np.ndarray, classification_params: ClassificationParams) -> List[int]:
    dim = classification_params.dim
    gamma_net_indices = []
    reshaped_prototypes_samples = prototypes_samples.reshape((-1, 1, dim))
    for prototype_sample_idx, prototype_sample in enumerate(reshaped_prototypes_samples):
        try:
            dist_to_gamma_net = np.min(np.linalg.norm(prototypes_samples[gamma_net_indices] - prototype_sample, axis=1))
        except ValueError:  # this occurs in case that the array is empty
            gamma_net_indices.append(prototype_sample_idx)
            continue
        if dist_to_gamma_net >= classification_params.gamma:
            gamma_net_indices.append(prototype_sample_idx)
    return gamma_net_indices


def get_prototypes_light_compressed_indices_and_labels_via_optinet(prototypes_samples: np.ndarray,
                                                                   train_samples: np.ndarray, train_labels: np.ndarray,
                                                                   classification_params: ClassificationParams) -> [
    Iterable, Iterable]:
    gamma_net_indices = get_gamma_net_indices(prototypes_samples, classification_params)
    # nn_tree = KNN(prototypes_samples[gamma_net_indices])
    # train_samples_nearest_prototype = nn_tree.query(train_samples, k=1)
    train_samples_nearest_prototype = query_k_nn(prototypes_samples[gamma_net_indices], train_samples, k=1)
    train_samples_nearest_prototype = train_samples_nearest_prototype.squeeze()
    queries_voters_indices_for_gamma_net = [
        np.where(train_samples_nearest_prototype == gamma_net_prototype_idx)[0]
        for gamma_net_prototype_idx in range(len(gamma_net_indices))
    ]
    gamma_net_majority_vote = get_majority_votes(queries_voters_indices_for_gamma_net, train_labels)
    return gamma_net_indices, gamma_net_majority_vote

def ver_3_get_prototypes_light_compressed_indices_and_weights_via_greedy_weighted_heuristic(
        train_samples: np.ndarray,
        train_labels: np.ndarray,
        classification_params: ClassificationParams,
        show_profile_stats: bool = False,
        ):
    '''
    algo with Roi and Lee-ad's theoretical optimization
    :param train_samples:
    :param train_labels:
    :param classification_params:
    :param show_profile_stats:
    :return:
    '''
    num_of_train_samples = len(train_samples)
    if num_of_train_samples <= 4:
        return np.arange(num_of_train_samples)
    # distances = np.zeros((num_of_train_samples,num_of_train_samples))
    # for p_inx in range(num_of_train_samples):
    #     print(p_inx)
    #     distances[p_inx][p_inx:] = np.linalg.norm([[train_samples[p_inx] - sample_b
    #                                         for sample_b in train_samples[p_inx:]]], axis=2)
    # distances = np.array(distances) + np.transpose(distances)
    # distances = np.linalg.norm(
    #     [[sample_b - sample_a for sample_b in train_samples] for sample_a in train_samples], axis=2)
    distances = classification_params.all_distances
    s_prime = np.ones(num_of_train_samples, dtype=bool)
    arg_sorted_indices = np.zeros([num_of_train_samples,num_of_train_samples], dtype='int')
    if len(distances)>0:
        arg_sorted_indices = np.argsort(distances)
    prototypes_indices = []
    d_ne = np.zeros(num_of_train_samples)
    idx_of_first_opposite_label_sample_indices = -np.ones(num_of_train_samples, dtype=int)
    counters = np.zeros(num_of_train_samples, dtype=int)
    for sample_idx in range(num_of_train_samples):
        if len(distances) > 0:
            cur_arg_sorted_indices = arg_sorted_indices[sample_idx]
        else:
            cur_arg_sorted_indices = np.argsort(compute_distances(train_samples[sample_idx],train_samples))
            arg_sorted_indices[sample_idx,:] = cur_arg_sorted_indices
        for neighbour in cur_arg_sorted_indices:
            if train_labels[sample_idx] == train_labels[neighbour]:
                counters[sample_idx] += 1
            else:
                if len(distances)>0:
                    d_ne[sample_idx] = distances[sample_idx, neighbour]
                else:
                    d_ne[sample_idx] = compute_distances(train_samples[sample_idx], [train_samples[neighbour]])
                idx_of_first_opposite_label_sample_indices[sample_idx] = counters[sample_idx]
                break

    safety_counter = num_of_train_samples + 1
    while np.any(s_prime):
        print(f'S_prime size = {np.sum(s_prime)}, counters_max = {counters.max()}')
        safety_counter -= 1
        if safety_counter <= 0:
            raise Exception(f'encountered an infinite loop')
        p_idx = np.argmax(counters)
        counters_max = counters.max()
        if counters_max == 1:
            where = np.where(s_prime == True)[0]
            prototypes_indices.extend(where)
            break
        if counters_max == 0:
            breakpoint()
            warn('counters max is 0')
        s_prime[
            arg_sorted_indices[p_idx][:idx_of_first_opposite_label_sample_indices[p_idx]]
        ] = False
        prototypes_indices.append(p_idx)
        # counters = update_counters_inball_and_s_prime(num_of_train_samples, train_labels, arg_sorted_indices, s_prime, idx_of_first_opposite_label_sample_indices)
        # counters = np.zeros(num_of_train_samples, dtype=int)
        for sample_idx in range(num_of_train_samples):
            if counters[sample_idx] == 0:
                continue
            current_counter = np.sum(s_prime[arg_sorted_indices[sample_idx, :idx_of_first_opposite_label_sample_indices[sample_idx]]])
            if current_counter == counters_max:
                 counters[sample_idx] = current_counter
                 if sample_idx != num_of_train_samples-1:
                     counters[sample_idx+1:] = -1
                 break
            # counters[sample_idx] = np.sum(s_prime[arg_sorted_indices[sample_idx, :idx_of_first_opposite_label_sample_indices[sample_idx]]])
            counters[sample_idx] = current_counter
    prototypes_indices = np.sort(prototypes_indices)
    weights = d_ne[prototypes_indices]
    return prototypes_indices, weights

def get_prototypes_light_compressed_indices_and_labels_via_rss(train_samples, train_labels, classification_params):
    distances = classification_params.all_distances
    num_of_train_samples = len(train_samples)
    arg_sorted_indices = np.zeros([num_of_train_samples, num_of_train_samples], dtype='int')
    if len(distances) > 0:
        arg_sorted_indices = np.argsort(distances)
    d_ne = np.zeros(num_of_train_samples)
    idx_of_first_opposite_label_sample_indices = -np.ones(num_of_train_samples, dtype=int)
    counters = np.zeros(num_of_train_samples, dtype=int)
    for sample_idx in range(num_of_train_samples):
        if len(distances) > 0:
            cur_arg_sorted_indices = arg_sorted_indices[sample_idx]
        else:
            cur_arg_sorted_indices = np.argsort(compute_distances(train_samples[sample_idx],train_samples))
        for neighbour in cur_arg_sorted_indices:
            if train_labels[sample_idx] == train_labels[neighbour]:
                counters[sample_idx] += 1
            else:
                if len(distances)>0:
                    d_ne[sample_idx] = distances[sample_idx, neighbour]
                else:
                    d_ne[sample_idx] = compute_distances(train_samples[sample_idx], [train_samples[neighbour]])
                idx_of_first_opposite_label_sample_indices[sample_idx] = counters[sample_idx]
                break
    samples_indices_sorted_by_d_ne = np.argsort(d_ne)
    prototype_indices = [samples_indices_sorted_by_d_ne[0]]
    for sample_inx in samples_indices_sorted_by_d_ne[1:]:
        if len(distances)>0:
            min_dist = np.min(distances[sample_inx,prototype_indices])
        else:
            min_dist = np.min(compute_distances(train_samples[sample_inx], train_samples[prototype_indices]))
        if min_dist >= d_ne[sample_inx]:
            prototype_indices.append(sample_inx)
    prototype_labels = train_labels[prototype_indices]
    return prototype_indices, prototype_labels


def get_prototypes_light_compressed_indices_and_labels_via_mss(train_samples, train_labels, classification_params):
    distances = classification_params.all_distances
    num_of_train_samples = len(train_samples)
    arg_sorted_indices = np.zeros([num_of_train_samples, num_of_train_samples], dtype='int')
    if len(distances) > 0:
        arg_sorted_indices = np.argsort(distances)
    d_ne = np.zeros(num_of_train_samples)
    idx_of_first_opposite_label_sample_indices = -np.ones(num_of_train_samples, dtype=int)
    counters = np.zeros(num_of_train_samples, dtype=int)
    for sample_idx in range(num_of_train_samples):
        if len(distances) > 0:
            cur_arg_sorted_indices = arg_sorted_indices[sample_idx]
        else:
            cur_arg_sorted_indices = np.argsort(compute_distances(train_samples[sample_idx],train_samples))
        for neighbour in cur_arg_sorted_indices:
            if train_labels[sample_idx] == train_labels[neighbour]:
                counters[sample_idx] += 1
            else:
                if len(distances) > 0:
                    d_ne[sample_idx] = distances[sample_idx, neighbour]
                else:
                    d_ne[sample_idx] = compute_distances(train_samples[sample_idx], [train_samples[neighbour]])
                idx_of_first_opposite_label_sample_indices[sample_idx] = counters[sample_idx]
                break
    samples_indices_sorted_by_d_ne = np.argsort(d_ne)
    s_prime = np.ones(num_of_train_samples, dtype=bool)
    prototype_indices = [samples_indices_sorted_by_d_ne[0]]
    for sample_inx in samples_indices_sorted_by_d_ne[1:]:
        add = False
        for sample_b in samples_indices_sorted_by_d_ne[sample_inx:]:
            if (s_prime[sample_b] and compute_distances(train_samples[sample_inx],[train_samples[sample_b]]) < d_ne[sample_b]):
                s_prime[sample_b] = False
                add = True
        if add:
            prototype_indices.append(sample_inx)
    prototype_labels = train_labels[prototype_indices]
    return prototype_indices, prototype_labels


def get_prototypes_light_compressed_indices_and_labels(prototypes_samples: np.ndarray, train_samples: np.ndarray,
                                                       train_labels: np.ndarray,
                                                       classification_params: ClassificationParams) -> [Iterable,
                                                                                                        Iterable,
                                                                                                        float]:
    classification_scheme = classification_params.classification_scheme
    if classification_scheme in [
        ClassificationSchemes.ONE_NN,
        ClassificationSchemes.K_NN
    ]:
        raise Exception(f'The classification methods {classification_scheme} does not change its prototypes\' labels')
    start_time = time()
    if classification_scheme == ClassificationSchemes.PROTO_K_NN:
        prototypes_algo_indices = np.arange(len(prototypes_samples))
        prototypes_algo_labels = get_proto_labels_of_proto_k_nn(prototypes_samples, train_samples, train_labels,
                                                                classification_params)
    elif classification_scheme == ClassificationSchemes.OPTINET:
        prototypes_algo_indices, prototypes_algo_labels = get_prototypes_light_compressed_indices_and_labels_via_optinet(
            prototypes_samples, train_samples, train_labels, classification_params)
    elif classification_scheme == ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC:
        prototypes_algo_indices, weights = ver_3_get_prototypes_light_compressed_indices_and_weights_via_greedy_weighted_heuristic(
            train_samples, train_labels, classification_params )
        prototypes_algo_labels = train_labels[prototypes_algo_indices]
        classification_params.set_nearest_neighbours_weights(weights)
    elif classification_scheme == ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED:
        prototypes_algo_indices, weights = get_prototypes_light_compressed_indices_and_weights_via_greedy_weighted_heuristic_reduced(
            train_samples, train_labels, classification_params)
        prototypes_algo_labels = train_labels[prototypes_algo_indices]
        classification_params.set_nearest_neighbours_weights(weights)
    elif classification_scheme == ClassificationSchemes.RSS:
        prototypes_algo_indices, prototypes_algo_labels = get_prototypes_light_compressed_indices_and_labels_via_rss(train_samples, train_labels, classification_params)
    elif classification_scheme == ClassificationSchemes.MSS:
        prototypes_algo_indices, prototypes_algo_labels = get_prototypes_light_compressed_indices_and_labels_via_mss(
            train_samples, train_labels, classification_params)
    else:
        raise Exception(f'classification_scheme {classification_scheme} is not supported')
    classification_preprocess_time = time() - start_time
    return prototypes_algo_indices, prototypes_algo_labels, classification_preprocess_time


def preprocess_classification_method(cur_train_samples: np.ndarray, cur_train_labels: np.ndarray,
                                     classification_params: ClassificationParams, m: Union[None, int]):
    if classification_params.should_draw_m():
        prototypes_samples = cur_train_samples[:m]
    else:
        prototypes_samples = cur_train_samples

    if classification_params.should_be_preprocessed():
        prototypes_light_compressed_indices, prototypes_light_compressed_algo_labels, classification_preprocess_time = get_prototypes_light_compressed_indices_and_labels(
            prototypes_samples, cur_train_samples, cur_train_labels, classification_params, )
        prototypes_samples = prototypes_samples[prototypes_light_compressed_indices]
    else:
        prototypes_light_compressed_algo_labels = cur_train_labels
        classification_preprocess_time = 0.0
    return prototypes_samples, prototypes_light_compressed_algo_labels, classification_preprocess_time


def infer_samples(prototypes_samples: np.ndarray, prototypes_labels: np.ndarray,
                  classification_params: ClassificationParams, query_samples: np.ndarray) -> [Iterable, float]:
    classification_scheme = classification_params.classification_scheme
    start_time = time()
    if classification_scheme in [
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED
    ]:
        weights = classification_params.nearest_neighbours_weights
        nn_indices = query_k_nn(prototypes_samples, query_samples, k=1, weights=weights)
        query_empirical_labels = prototypes_labels[nn_indices]
    else:
        if classification_scheme in [
            ClassificationSchemes.ONE_NN,
            ClassificationSchemes.PROTO_K_NN,
            ClassificationSchemes.OPTINET,
            ClassificationSchemes.RSS,
            ClassificationSchemes.MSS
        ]:
            acting_k = 1
        elif classification_scheme == ClassificationSchemes.K_NN:
            acting_k = classification_params.k
        else:
            raise Exception(f'Unsupported classification scheme {classification_scheme}')
        if len(prototypes_samples) <= acting_k:  # this is an extreme case where k is greater than the number of prototypes samples
            label = get_majority_votes([np.arange(len(prototypes_labels))], prototypes_labels)[0]
            query_empirical_labels = [label] * len(query_samples)
        else:
            # tree = KNN(prototypes_samples)
            # nearest_prototypes = tree.query(query_samples, acting_k)
            nearest_prototypes = query_k_nn(prototypes_samples, query_samples, k=acting_k)
            if acting_k == 1:
                query_empirical_labels = prototypes_labels[nearest_prototypes.squeeze()]
            else:
                query_empirical_labels = get_majority_votes(nearest_prototypes, prototypes_labels)
    inference_time = time() - start_time
    return query_empirical_labels, inference_time

    def verify_consistency_on_train_sample():
        return
