from __future__ import annotations
from collections.abc import Iterable
from typing import Union, Literal, List

import numpy as np

from _constants import ALL, CompressionSchemes, ClassificationSchemes

class ClassificationParams:
    nearest_neighbours_weights: Union[Literal['uniform'], np.ndarray]
    all_distances: np.ndarray
    # d_nes: np.ndarray

    def __init__(self, classification_scheme: ClassificationSchemes, dim: int, gamma: Union[None, float] = None, k: Union[None, int] = None):
        self.classification_scheme = classification_scheme
        self.dim = dim
        self.gamma = gamma
        self.k = k
        self.identifier = f'{self.classification_scheme.value}'

        self.nearest_neighbours_weights = 'uniform'  # 'uniform' is the default weights of knn sklearn

    def set_nearest_neighbours_weights(self, weights: np.ndarray):
        self.nearest_neighbours_weights = weights

    def set_all_distances(self, all_distances:np.ndarray):
        self.all_distances = all_distances

    # def set_d_nes(self, d_nes):
    #     self.d_nes = d_nes

    def should_draw_m(self) -> bool:
        if self.classification_scheme in [
            ClassificationSchemes.PROTO_K_NN,
            # ClassificationSchemes.OPTINET
        ]:
            return True
        if self.classification_scheme in [
            ClassificationSchemes.ONE_NN,
            ClassificationSchemes.K_NN,
            ClassificationSchemes.OPTINET,
            ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
            ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED,
            ClassificationSchemes.RSS,
            ClassificationSchemes.MSS
        ]:
            return False
        raise Exception(f'classification scheme {self.classification_scheme} is not supported')

    def does_use_hyper_param_k(self):
        if self.classification_scheme in [
            ClassificationSchemes.K_NN,
            ClassificationSchemes.PROTO_K_NN,
            ClassificationSchemes.OPTINET,
        ]:
            return True
        if self.classification_scheme in [
            ClassificationSchemes.ONE_NN
        ]:
            return False
        raise Exception(f'classification scheme {self.classification_scheme} is not supported')

    def does_use_hyper_param_gamma(self):
        if self.classification_scheme in [
            ClassificationSchemes.PROTO_K_NN,
            ClassificationSchemes.OPTINET,
        ]:
            return True
        if self.classification_scheme in [
            ClassificationSchemes.ONE_NN,
            ClassificationSchemes.K_NN,
        ]:
            return False
        raise Exception(f'classification scheme {self.classification_scheme} is not supported')

    def should_be_preprocessed(self):
        if self.classification_scheme in [
            ClassificationSchemes.PROTO_K_NN,
            ClassificationSchemes.OPTINET,
            ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
            ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED,
            ClassificationSchemes.RSS,
            ClassificationSchemes.MSS
        ]:
            return True
        if self.classification_scheme in [
            ClassificationSchemes.ONE_NN,
            ClassificationSchemes.K_NN,
        ]:
            return False
        raise Exception(f'classification scheme {self.classification_scheme} is not supported')

class CompressionParams:
    def __init__(self, compression_scheme: CompressionSchemes, dim: int):
        self.compression_scheme = compression_scheme
        self.dim = dim
        self.identifier = f'{self.compression_scheme.value}'


class UserParams:
    user_amount_of_samples: int
    samples_amounts_list: List[int]
    training_percent: float
    num_of_iterations: int

    compression_schemes: Iterable
    gamma: float
    classification_schemes: Iterable

    def __init__(
            self,
            num_of_iterations: int,
            total_num_of_samples: Union[int, Literal[ALL]],
            num_of_samples_amounts: int,
            training_percent: float,
            classification_schemes: Union[Iterable, Literal[ALL]],
            gamma: float,
            k: int,
            compression_schemes: Union[Iterable, None, Literal[ALL]]
    ):
        self.user_amount_of_samples = 18720 if total_num_of_samples == ALL else total_num_of_samples
        # if self.user_amount_of_samples > 18720:
        #     raise Exception(f'available data samples per class is 18720, but {total_num_of_samples} were requested')
        self.num_of_samples_amounts = num_of_samples_amounts
        self.training_percent = training_percent
        self.num_of_iterations = num_of_iterations
        self.compression_schemes = [] if compression_schemes == ALL else compression_schemes
        self.gamma = gamma
        self.k = k
        self.classification_schemes = classification_schemes

    def get_classification_params_list(self, dim: int) -> List[ClassificationParams]:
        return [ClassificationParams(classification_scheme, dim, self.gamma, self.k) for classification_scheme in self.classification_schemes]

    def get_compression_params_list(self, dim: int) -> List[CompressionParams]:
        return [CompressionParams(compression_scheme, dim) for compression_scheme in self.compression_schemes]

    def get_samples_amounts_list(self, total_num_of_train_samples):
        acting_num_of_samples = min(total_num_of_train_samples, self.user_amount_of_samples)
        self.samples_amounts_list = [int(i * (acting_num_of_samples / self.num_of_samples_amounts)) for i in range(1, self.num_of_samples_amounts + 1)]
        return self.samples_amounts_list

