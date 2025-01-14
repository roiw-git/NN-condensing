import json
from collections import defaultdict
from copy import deepcopy
from typing import List, Union

import numpy as np

from params.program_params.program_params_handler import ProgramParams


def recursively_default_dict(added_dictionary: Union[dict, None] = None):
    if added_dictionary is not None:
        return defaultdict(recursively_default_dict, added_dictionary)
    return defaultdict(recursively_default_dict)


class ResultsCalculationHandler:
    NUM_OF_SAMPLES: str = 'num_of_samples'
    TIMES: str = 'times'
    ERROR: str = 'error'
    CLASSIFICATION: str = 'classification'
    COMPRESSION: str = 'compression'
    PREPROCESS: str = 'preprocess'
    INFERENCE: str = 'inference'

    data: defaultdict

    def __init__(self, num_of_iterations: int):
        self.data = recursively_default_dict()

    def set_classification_preprocess_time(self, classification_identifier: str, classification_preprocess_time: float,
                                           iteration_idx: int, num_of_samples: int) -> None:
        self.data[iteration_idx][num_of_samples][self.TIMES][classification_identifier][
            self.PREPROCESS] = classification_preprocess_time
        return

    def update_models_results(self, test_labels: np.ndarray, empirical_test_labels: np.ndarray,
                              compressed_data_size: int, iteration_idx: int,
                              num_of_samples: int, classification_identifier: str, compression_identifier: str,
                              compression_time: float, inference_time: float):
        self.data[iteration_idx][num_of_samples][self.TIMES][classification_identifier][self.COMPRESSION][
            compression_identifier] = compression_time
        self.data[iteration_idx][num_of_samples][self.TIMES][classification_identifier][self.INFERENCE][
            compression_identifier] = inference_time
        self.data[iteration_idx][num_of_samples][self.COMPRESSION][classification_identifier][
            compression_identifier] = compressed_data_size / num_of_samples
        error_prob = self.calc_error(test_labels, empirical_test_labels)
        self.data[iteration_idx][num_of_samples][self.ERROR][classification_identifier][
            compression_identifier] = error_prob
        if error_prob >= 0.99:
            # breakpoint()
            pass

    @staticmethod
    def calc_error(labels_set_1: np.ndarray, labels_set_2: np.ndarray) -> float:
        return np.sum(labels_set_1 != labels_set_2) / len(labels_set_1)

    def store_results(self, program_params: ProgramParams) -> None:
        if not program_params.store_results:
            return
        self.calc_mean_results()
        out_path_dir = program_params.out_path_dir
        if not out_path_dir.is_dir():
            out_path_dir.mkdir()
        with open(out_path_dir / 'results.json', 'w') as f:
            json.dump(self.mean_results, f, indent=4, ensure_ascii=True)
        return

    def re_calc_mean_results(self, _dict_obj: Union[None, dict] = None, safety_integer: int = 4000):
        if safety_integer <= 0:
            raise Exception(f'possibly encountered an infinite loop')
        safety_integer -= 1
        if _dict_obj is None:
            _dict_obj = self.data
            # for iteration_num in self.data.keys():

    def calc_mean_results(self) -> dict:
        for iter_idx, iter_results in enumerate(self.data.values()):
            if iter_idx == 0:
                accumulative_dict = deepcopy(iter_results)
                continue
            self.add_to_accumulative_dict(accumulative_dict, iter_results)
        num_of_iterations = len(self.data)
        self.divide_accumulative_in_num_of_iterations(accumulative_dict, num_of_iterations=num_of_iterations)
        accumulative_dict = self.convert_default_dict_to_dict(accumulative_dict)
        self.mean_results = accumulative_dict
        return self.mean_results

    @classmethod
    def add_to_accumulative_dict(cls, cur_accumulative_dict: dict, cur_results: dict, safety_integer: int = 4000) -> None:
        if safety_integer <= 0:
            raise Exception(f'possibly encountered an infinite loop')
        safety_integer -= 1
        for key, result_value in cur_results.items():
            if isinstance(result_value, float):
                cur_accumulative_dict[key] += result_value
            elif isinstance(result_value, dict):
                cls.add_to_accumulative_dict(cur_accumulative_dict[key], result_value, safety_integer)
            else:
                raise Exception(f'value type should be Union[dict, float], found: {type(result_value)}')
        return

    @classmethod
    def divide_accumulative_in_num_of_iterations(cls, cur_accumulative_dict: dict, num_of_iterations: int, safety_integer: int = 4000) -> None:
        if safety_integer <= 0:
            raise Exception(f'possibly encountered an infinite loop')
        safety_integer -= 1
        for key, value in cur_accumulative_dict.items():
            if isinstance(value, float):
                cur_accumulative_dict[key] = value / num_of_iterations
            elif isinstance(value, dict):
                cls.divide_accumulative_in_num_of_iterations(value, num_of_iterations, safety_integer)
            else:
                raise Exception(f'value type should be Union[dict, float], found: {type(value)}')
        return None

    @classmethod
    def convert_default_dict_to_dict(cls, default_dict: defaultdict, safety_integer: int = 4000) -> dict:
        if safety_integer <= 0:
            raise Exception(f'possibly encountered an infinite loop')
        safety_integer -= 1
        dictionary = dict(default_dict)
        for key, value in dictionary.items():
            if isinstance(value, defaultdict):
                dictionary[key] = dict(value)
                cls.convert_default_dict_to_dict(value, safety_integer)
        return dictionary








