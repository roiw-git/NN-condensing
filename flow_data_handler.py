import pickle

import numpy as np
from enum import Enum, auto

from _constants import ROOT_PATH, SEED
from params.user_params.user_params_handler import UserParams
import csv
import pandas as pd
from ast import literal_eval

class Types(Enum):
    TRAIN_DATA = auto()
    TEST_DATA = auto()
    ALL_DATA = auto()

# def get_raw_dataset(user_params: UserParams) -> [np.ndarray, np.ndarray]:
#     if user_params.dataset == 'notMNIST':
#         return get_raw_dataset_notMNIST(user_params)
#     elif user_params.dataset == 'magic':
#         return get_raw_dataset_magic(user_params)
#     elif user_params.dataset == 'satimage':
#         return get_raw_dataset_magic(user_params)

def get_raw_dataset_notMNIST(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    # np.random.seed(SEED)
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'notMNIST' / 'umap_3_d.p'
    with open(data_pickle_path, 'rb') as f:
        raw_data = pickle.load(f)
    train_amounts = {key: int(len(value)) for key, value in raw_data.items()}
    ordered_keys = list(raw_data.keys())  # in order to keep samples to labels compatibility
    dataset = {key: value for key, value in raw_data.items()}
    samples = np.concatenate([dataset[key] for key in ordered_keys])
    labels = np.concatenate([[key] * len(dataset[key]) for key in ordered_keys])
    return samples, labels

def get_raw_dataset_magic(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'magic' / 'magic04.data'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:10]] for result in points])
        labels = np.concatenate([[result[10]] for result in points])
        return samples, labels

def get_raw_dataset_satimage(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'satimage' / 'satimage+refactor.csv'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:36]] for result in points])
        labels = np.concatenate([[result[36]] for result in points])
        return samples, labels

def get_raw_dataset_spambase(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'spambase' / 'spambase_refactor.data'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:57]] for result in points])
        labels = np.concatenate([[result[57]] for result in points])
        return samples, labels

def get_raw_dataset_twonorm(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'twonorm' / 'twonorm+refactor.data'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:20]] for result in points])
        labels = np.concatenate([[result[20]] for result in points])
        return samples, labels

def get_raw_dataset_phoneme(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'phoneme' / 'phoneme_refactored.csv'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:5]] for result in points])
        labels = np.concatenate([[result[5]] for result in points])
        return samples, labels

def get_raw_dataset_segment(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'segment' / 'segment_refactored.csv'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = csv.reader(f, delimiter='!')
        points = []
        for row in data:
            result = literal_eval(row[0])
            # points.extend(result)
            # samples = np.append([result])
            points.append(result)
        samples = np.concatenate([[result[0:19]] for result in points])
        labels = np.concatenate([[result[19]] for result in points])
        return samples, labels

def get_raw_dataset_cifar_10(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'cifar_10' / 'cifar_10.txt'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = pd.read_csv(f, delimiter='!', nrows=49998)
        data = pd.DataFrame.to_numpy(data)
        # data = csv.reader(f, delimiter='!')
        points = []
        i=0
        for row in data:
            print(i); i+=1
            result = literal_eval(row[0])
            if result[0][3072] in [1,2,3,4]:
                points.append(result)
        samples = np.concatenate([result[0:3072] for result in points])
        labels = np.concatenate([[result[0][3072]] for result in points])
        return samples, labels

def get_raw_dataset_shuttle(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'shuttle' / 'shuttle_trn.csv'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = pd.read_csv(f, delimiter='!')
        data = pd.DataFrame.to_numpy(data)
        # data = csv.reader(f, delimiter='!')
        points = []
        i=0
        for row in data:
            print(i); i+=1
            result = literal_eval(row[0])
            points.append(result)
        samples = np.concatenate([[result[0:9]] for result in points])
        labels = np.concatenate([[result[9]] for result in points])
        return samples, labels

def get_raw_dataset_adult(user_params: UserParams) -> [np.ndarray, np.ndarray]:
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'adult' / 'adult.csv'
    with open(data_pickle_path, newline='') as f:
        # raw_data = pickle.load(f)
        data = pd.read_csv(f, delimiter='!')
        data = pd.DataFrame.to_numpy(data)
        # data = csv.reader(f, delimiter='!')
        points = []
        i=0
        for row in data:
            print(i); i+=1
            result = literal_eval(row[0])
            points.append(result)
        samples = np.concatenate([[result[0:6]] for result in points])
        labels = np.concatenate([[result[14]] for result in points])
        return samples, labels

def shuffle_and_split_dataset(samples, labels, user_params: UserParams):
    training_percent = user_params.training_percent
    training_amount = int(len(samples) * training_percent)
    random_permutation = np.random.permutation(np.arange(len(samples)))
    samples = samples[random_permutation]
    labels = labels[random_permutation]
    train_samples = samples[:training_amount]
    train_labels = labels[:training_amount]
    test_samples = samples[training_amount:]
    test_labels = labels[training_amount:]
    return train_samples, train_labels, test_samples, test_labels

def shuffle_and_split_dataset_with_distances(samples, labels, distances, user_params: UserParams):
    training_percent = user_params.training_percent
    training_amount = int(len(samples) * training_percent)
    random_permutation = np.random.permutation(np.arange(len(samples)))
    samples = samples[random_permutation]
    labels = labels[random_permutation]
    distances = distances[random_permutation,:]
    distances = distances[:,random_permutation]
    train_samples = samples[:training_amount]
    train_labels = labels[:training_amount]
    test_samples = samples[training_amount:]
    test_labels = labels[training_amount:]
    train_distances = distances[:training_amount, :training_amount]
    return train_samples, train_labels, test_samples, test_labels, train_distances

def shuffle_and_split_dataset(samples, labels, user_params: UserParams):
    training_percent = user_params.training_percent
    training_amount = int(len(samples) * training_percent)
    random_permutation = np.random.permutation(np.arange(len(samples)))
    samples = samples[random_permutation]
    labels = labels[random_permutation]
    # distances = distances[random_permutation,:]
    # distances = distances[:,random_permutation]
    train_samples = samples[:training_amount]
    train_labels = labels[:training_amount]
    test_samples = samples[training_amount:]
    test_labels = labels[training_amount:]
    # train_distances = distances[:training_amount, :training_amount]
    return train_samples, train_labels, test_samples, test_labels


def get_datasets(user_params: UserParams) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # np.random.seed(SEED)
    data_pickle_path = ROOT_PATH / 'resources' / 'in' / 'umap_3_d.p'
    with open(data_pickle_path, 'rb') as f:
        raw_data = pickle.load(f)
    training_percent = user_params.training_percent
    train_amounts = {key: int(len(value) * training_percent) for key, value in raw_data.items()}
    ordered_keys = list(raw_data.keys())  # in order to keep samples to labels compatibility

    train_data = {key: value[:train_amounts[key]] for key, value in raw_data.items()}
    test_data = {key: value[train_amounts[key]:] for key, value in raw_data.items()}

    train_samples = np.concatenate([train_data[key] for key in ordered_keys])
    train_labels = np.concatenate([[key] * len(train_data[key]) for key in ordered_keys])
    indices = np.arange(len(train_samples))
    np.random.shuffle(indices)
    train_samples = train_samples[indices]
    train_labels = train_labels[indices]

    test_samples = np.concatenate([test_data[key] for key in ordered_keys])
    test_labels = np.concatenate([[key] * len(test_data[key]) for key in ordered_keys])

    return train_samples, train_labels, test_samples, test_labels
