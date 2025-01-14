import numpy as np
import winsound
import cProfile, pstats, io
from pstats import SortKey
import pickle

from _constants import SEED, CompressionSchemes, ClassificationSchemes
from compression import get_compressed_indices
from flow_data_handler import *
# from flow_data_handler import get_datasets, shuffle_and_split_dataset, shuffle_and_split_dataset_with_distances, get_raw_dataset_magic, get_raw_dataset_satimage, get_raw_dataset_spambase, get_raw_dataset_twonorm, get_raw_dataset_phoneme, get_raw_dataset_segment,get_raw_dataset_cifar_10
from params.plots_params.plots_params import plots_params_dict
from params.plots_params.plots_params_handler import PlotsParams
from params.program_params.program_params import program_params_dict
from params.program_params.program_params_handler import ProgramParams
from params.user_params.user_params import user_params_dict
from params.user_params.user_params_handler import UserParams, ClassificationParams, CompressionParams
from classification import infer_samples, preprocess_classification_method
from plots import make_plots
from results_calculation_manager import ResultsCalculationHandler
from results_storage import FlowInfoForStorage


def reset_seed():
    np.random.seed(SEED)


def calculate_m(num_of_samples, k):
    m = int(num_of_samples / k)
    if m <= 0:
        m = 1
    elif m > num_of_samples:
        m = num_of_samples
    return m


def run_combination(classification_params: ClassificationParams, compression_params: CompressionParams):
    classification_scheme = classification_params.classification_scheme
    compression_scheme = compression_params.compression_scheme
    if classification_scheme in [
        ClassificationSchemes.K_NN,
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED,
        # ClassificationSchemes.RSS,
    ]:
        if compression_scheme == CompressionSchemes.NONE:
            return True
        else:
            return False
    else:
        if compression_scheme != CompressionSchemes.PROTOAPPROX:
            return True
        else:
            return False

def does_use_all_distances_computation(classification_params: ClassificationParams):
    classification_scheme = classification_params.classification_scheme
    if classification_scheme in [
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
        ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED,
        ClassificationSchemes.RSS,
        ClassificationSchemes.MSS,
    ]:
        return True
    else:
        return False

def store_all_distances(all_distances, program_params) -> None:
    out_path_dir = program_params.out_path_dir
    with open(out_path_dir / 'all_points_distances', 'wb') as f:
        pickle.dump(all_distances, f)
    return

def load_all_distances(program_params):
    with open(program_params.out_path_dir / 'all_points_distances', 'rb') as f:
        str_results = pickle.load(f)
    return str_results

def run_flow():
    # run_new_experiment
    program_params = ProgramParams(**program_params_dict)
    user_params = UserParams(**user_params_dict)
    reset_seed()
    if program_params.run_new_experiment:
        show_in_run_stats = program_params.show_in_run_stats
        # all_samples, all_labels = get_raw_dataset_adult(user_params)
        # all_samples, all_labels = get_raw_dataset_shuttle(user_params)
        # all_samples, all_labels = get_raw_dataset_cifar_10(user_params)
        all_samples, all_labels = get_raw_dataset_segment(user_params)
        # all_samples, all_labels = get_raw_dataset_phoneme(user_params)
        # all_samples, all_labels = get_raw_dataset_twonorm(user_params)
        # all_samples, all_labels = get_raw_dataset_spambase(user_params)
        # all_samples, all_labels = get_raw_dataset_satimage(user_params)
        # all_samples, all_labels = get_raw_dataset_magic(user_params)
        # all_samples, all_labels = get_raw_dataset_notMNIST(user_params)
        dim = all_samples.shape[1]
        samples_amounts_list = user_params.get_samples_amounts_list(total_num_of_train_samples= int(len(all_samples) * user_params.training_percent))
        if program_params.compute_all_distances_first:
            print(f'Computing all point distances')
            # max_num_of_train_samples = np.max(samples_amounts_list)
            all_points_distances = np.zeros((len(all_samples), len(all_samples)))
            for p_inx in range(len(all_samples)):
                print(p_inx)
                all_points_distances[p_inx][p_inx:] = np.linalg.norm([[all_samples[p_inx] - sample_b
                                                                       for sample_b in all_samples[
                                                                                       p_inx:len(all_samples)]]],
                                                                     axis=2)
            all_points_distances = np.array(all_points_distances) + np.transpose(all_points_distances)
            store_all_distances(all_points_distances, program_params)
        elif program_params.load_all_distances:
            all_points_distances = load_all_distances(program_params)
        else:
            all_points_distances=[]
        # all_train_samples, all_train_labels, test_samples, test_labels = get_datasets(user_params)
        # dim = all_train_samples.shape[1]
        # samples_amounts_list = user_params.get_samples_amounts_list(total_num_of_train_samples=len(all_train_samples))
        classification_params_list = user_params.get_classification_params_list(dim=dim)
        compression_params_list = user_params.get_compression_params_list(dim=dim)

        num_of_iterations = user_params.num_of_iterations
        iterations_results = ResultsCalculationHandler(num_of_iterations=num_of_iterations)

        if program_params.show_profile_stats:
            pr = cProfile.Profile()
            pr.enable()
        for iteration_idx in range(1, num_of_iterations + 1):
            if len(all_points_distances)>0:
                all_train_samples, all_train_labels, test_samples, test_labels, all_train_distances = shuffle_and_split_dataset_with_distances(all_samples, all_labels, all_points_distances, user_params)
            else:
                all_train_samples, all_train_labels, test_samples, test_labels = shuffle_and_split_dataset(all_samples, all_labels, user_params)
            # if program_params.compute_all_distances_first:
            #     print(f'Computing all point distances')
            #     max_num_of_train_samples = np.max(samples_amounts_list)
            #     all_points_distances = np.zeros((max_num_of_train_samples,max_num_of_train_samples))
            #     for p_inx in range(max_num_of_train_samples):
            #         print(p_inx)
            #         all_points_distances[p_inx][p_inx:] = np.linalg.norm([[all_train_samples[p_inx] - sample_b
            #                                 for sample_b in all_train_samples[p_inx:max_num_of_train_samples]]], axis=2)
            #     all_points_distances = np.array(all_points_distances) + np.transpose(all_points_distances)
            for num_of_samples_idx, num_of_samples in enumerate(samples_amounts_list):
                cur_train_samples = all_train_samples[:num_of_samples]
                cur_train_labels = all_train_labels[:num_of_samples]
                if program_params.load_all_distances or program_params.compute_all_distances_first:
                    cur_distances = all_train_distances[:num_of_samples, :num_of_samples]
                else:
                    cur_distances = []
                # if program_params.compute_all_distances_first:
                #     cur_distances = all_points_distances[:num_of_samples,:num_of_samples]
                # else:
                #     cur_distances = []
                m = calculate_m(num_of_samples, user_params.k)
                if iteration_idx == 1:
                    flow_info_for_storage = FlowInfoForStorage(all_train_samples, all_train_labels, test_samples,
                                                               test_labels)
                for classification_idx, classification_params in enumerate(classification_params_list):
                    classification_identifier = classification_params.identifier
                    # if classification_params.classification_scheme in [ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC, ClassificationSchemes.RSS]:
                    #     classification_params.set_all_distances(cur_distances)
                    if does_use_all_distances_computation(classification_params):
                        classification_params.set_all_distances(cur_distances)
                    if show_in_run_stats:
                        print(f'Started preprocess for classification {classification_identifier}')
                    prototypes_samples, prototypes_light_compressed_algo_labels, classification_preprocess_time = preprocess_classification_method(
                        cur_train_samples, cur_train_labels, classification_params, m)
                    prototypes_algo_labels = prototypes_light_compressed_algo_labels
                    iterations_results.set_classification_preprocess_time(classification_identifier,
                                                                          classification_preprocess_time, iteration_idx,
                                                                          num_of_samples)
                    for compression_idx, compression_params in enumerate(compression_params_list):
                        if not run_combination(classification_params, compression_params):
                            continue
                        if show_in_run_stats:
                            print(f'iter {iteration_idx} out of {num_of_iterations}\n'
                                  f'group sample {num_of_samples_idx + 1} / {len(user_params.samples_amounts_list)}\n'
                                  f'classification idx {classification_idx + 1} / {len(classification_params_list)} {classification_identifier}\n'
                                  f'compression idx {compression_idx + 1} / {len(compression_params_list)} {compression_params.identifier}\n\n')
                        compression_identifier = compression_params.identifier
                        compressed_indices, compression_time = get_compressed_indices(prototypes_samples,
                                                                                      prototypes_algo_labels,
                                                                                      cur_train_samples,
                                                                                      compression_params)
                        compressed_prototypes_samples = prototypes_samples[compressed_indices]
                        compressed_prototypes_algo_labels = prototypes_algo_labels[
                            compressed_indices]
                        empirical_test_labels, inference_time = infer_samples(compressed_prototypes_samples,
                                                                              compressed_prototypes_algo_labels,
                                                                              classification_params,
                                                                              test_samples)
                        compressed_data_size = len(compressed_indices)
                        iterations_results.update_models_results(test_labels, empirical_test_labels, compressed_data_size,
                                                                 iteration_idx, num_of_samples,
                                                                 classification_identifier, compression_identifier,
                                                                 compression_time,
                                                                 inference_time)
                        if iteration_idx == 1:
                            flow_info_for_storage.add_flow_info(num_of_samples, classification_identifier,
                                                                compression_identifier, compressed_indices,
                                                                compressed_prototypes_algo_labels,
                                                                empirical_test_labels)
        if program_params.show_profile_stats:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.TIME
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        iterations_results.store_results(program_params)
        flow_info_for_storage.store_results(program_params)
    plots_params = PlotsParams(**plots_params_dict)
    make_plots(program_params, plots_params)
    return


if __name__ == '__main__':
    run_flow()
    # frequency = 2500
    # duration = 1300
    # winsound.Beep(frequency, duration)
    print('Done')
