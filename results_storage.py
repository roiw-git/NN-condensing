import json
import pickle
from collections import defaultdict

import numpy as np


def recursively_default_dict():
    return defaultdict(recursively_default_dict)


class FlowInfoForStorage:
    DATA: str = 'data'
    TRAIN: str = 'train'
    TEST: str = 'test'
    SAMPLES: str = 'samples'
    LABELS: str = 'labels'
    NUM_OF_SAMPLES: str = 'num_of_samples'
    COMPRESSED_INDICES: str = 'compressed_indices'
    COMPRESSED_PROTOTYPES_EMPIRICAL_LABELS: str = 'compressed_prototypes_empirical_labels'
    EMPIRICAL_TEST_LABELS: str = 'empirical_test_labels'

    def __init__(self, all_train_samples: np.ndarray, all_train_labels: np.ndarray, test_samples: np.ndarray,
                 test_labels: np.ndarray):
        self.info = {
            self.DATA: {
                self.TRAIN: {
                    self.SAMPLES: all_train_samples,
                    self.LABELS: all_train_labels
                },
                self.TEST: {
                    self.SAMPLES: test_samples,
                    self.LABELS: test_labels
                }
            },
            self.NUM_OF_SAMPLES: {}
        }

    def add_flow_info(self, num_of_samples: int, classification_identifier: str, compression_identifier: str,
                      compressed_indices: np.ndarray, compressed_prototypes_empirical_labels: np.ndarray,
                      empirical_test_labels: np.ndarray):
        self.info[self.NUM_OF_SAMPLES][num_of_samples] = {
            classification_identifier: {
                compression_identifier: {
                    self.COMPRESSED_INDICES: compressed_indices,
                    self.COMPRESSED_PROTOTYPES_EMPIRICAL_LABELS: compressed_prototypes_empirical_labels,
                    self.EMPIRICAL_TEST_LABELS: empirical_test_labels
                }
            }
        }
        return

    def store_results(self, program_params) -> None:
        out_path_dir = program_params.out_path_dir
        with open(out_path_dir / 'prototypes_info_of_one_iteration.pkl', 'wb') as f:
            pickle.dump(self.info, f)
        return

