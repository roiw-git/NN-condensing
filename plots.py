import json
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from _constants import legend_dict

from _constants import CompressionSchemes, ClassificationSchemes
from params.plots_params.plots_params_handler import PlotsParams
from params.program_params.program_params_handler import ProgramParams

def load_results(program_params: ProgramParams) -> dict:
    with open(program_params.out_path_dir / 'results.json', 'r') as f:
        str_results = json.load(f)
    # convert str keys to ints
    results = {}
    for key, value in str_results.items():
        if str.isdigit(key):
            results[int(key)] = value
        else:
            results[key] = value
    samples_sizes = np.array(list(results))
    classifications_schemes = list(list(results.values())[0]['error'])
    compression_schemes = list(list(list(results.values())[0]['error'].values())[0])
    return results, classifications_schemes, compression_schemes, samples_sizes

def should_add_to_plot(classification_scheme, compression_scheme):
    if classification_scheme in ['greedy_weighted_heuristic', 'greedy_weighted_heuristic_reduced']:
        if compression_scheme == 'none':
            return True
        return False
    if classification_scheme in ['rss', 'mss']:
        # if compression_scheme in ['none', 'protocomp']:
        if compression_scheme in ['none']: # changed to remove protocomp
            return True
        return False
    if classification_scheme == 'k_nn':
        if compression_scheme == 'none':
            return True
        return False
    if classification_scheme in [
        'one_nn',
        'proto_k_nn',
        'optinet'
    ]:
        # if compression_scheme in ['none', 'protocomp']:
        if compression_scheme in ['none']: # changed to remove protocomp
            return True
        else:
            return False
    return False

def should_plot_compression(classification_scheme, compress_scheme, large_compression_idx: int = 0):
    # if large_compression_idx > 0:
    #     if classification_scheme == 'one_nn' and compress_scheme == 'protocomp':
    #         return False
    # if large_compression_idx > 1:
    #     if classification_scheme in ['greedy_weighted_heuristic', 'greedy_weighted_heuristic_reduced']:
    #         return False
    if compress_scheme == 'protocomp':
        # return True
        return False # changed to remove protocomp
    if classification_scheme in ['optinet', 'proto_k_nn', 'greedy_weighted_heuristic', 'greedy_weighted_heuristic_reduced', 'rss', 'mss', 'one_nn']:
        return True
    return False

def make_plots(program_params: ProgramParams, plots_params: PlotsParams) -> None:
    if (not plots_params.bool_save_plots) and (not plots_params.bool_show_plots):
        return
    results, classifications_schemes, compression_schemes, samples_sizes = load_results(program_params)
    fig_times, ax_times = plt.subplots()
    fig_compression, ax_compression = plt.subplots()
    # fig_compression_of_large_scale_1, ax_compression_of_large_scale_1 = plt.subplots()
    # fig_compression_of_large_scale_2, ax_compression_of_large_scale_2 = plt.subplots()
    fig_error, ax_error = plt.subplots()

    # fig_times.suptitle('times')
    # fig_compression.suptitle('compression')
    # fig_error.suptitle('error')

    fontsize = 12

    which = None
    # 'minor'
    ax_times.grid(which)
    ax_compression.grid(which)
    # ax_compression_of_large_scale_1.grid(which)
    # ax_compression_of_large_scale_2.grid(which)
    ax_error.grid(which)

    num_of_models = len(list(filter(lambda model_combination: should_add_to_plot(*model_combination), product(classifications_schemes, compression_schemes))))
    min_ms = 5
    delta = 2.5
    # max_ms = 13
    # ms_iter = iter(np.arange(min_ms, max_ms+1, (max_ms - min_ms) / (num_of_models - 1))[::-1])
    ms_iter = iter([min_ms + delta*i for i in range(num_of_models)][::-1])
    for classification_scheme, compress_scheme in product(classifications_schemes, compression_schemes):
        if compress_scheme == 'none':
            # label = classification_scheme
            label = legend_dict[classification_scheme]

        else:
            label = f'{legend_dict[classification_scheme]} + {legend_dict[compress_scheme]}'

        if not should_add_to_plot(classification_scheme, compress_scheme):
            continue

        ms = next(ms_iter)

        ## times ##
        classification_preprocess_times = [result['times'][classification_scheme]['preprocess'] for result in results.values()]
        compression_times = [result['times'][classification_scheme]['compression'][compress_scheme] for result in results.values()]
        inference_times = [result['times'][classification_scheme]['inference'][compress_scheme] for result in results.values()]

        classification_preprocess_times = np.array(classification_preprocess_times)
        compression_times = np.array(compression_times)
        inference_times = np.array(inference_times)

        class_and_comp_times = classification_preprocess_times + compression_times + inference_times
        ax_times.plot(samples_sizes[1:], class_and_comp_times[1:], marker='o', ms=ms, label=label)
        # ax_times.set_xscale('log')
        # ax_times.set_yscale('log')
        ax_times.set_xlabel('Number of samples',fontsize = fontsize)
        ax_times.set_ylabel('Runtime (sec)',fontsize = fontsize)
        num_of_samp_ticks = [4000, 6000, 8000, 10000, 12000, 14000]
        # ax_times.set_xticks(num_of_samp_ticks, num_of_samp_ticks)
        ##########

        ##  compression  ##
        if should_plot_compression(classification_scheme, compress_scheme):
            compression_rate = [result['compression'][classification_scheme][compress_scheme] for result in results.values()]
            ax_compression.plot(samples_sizes[0:], compression_rate[0:], marker='o', ms=ms, label=label)
            # ax_compression.set_xscale('log')
            # ax_compression.set_yscale('log')
            ax_compression.set_xlabel('Number of samples',fontsize = fontsize)
            ax_compression.set_ylabel('Compression rate',fontsize = fontsize)
            comp_ticks = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
            # ax_compression.set_yticks(comp_ticks, comp_ticks)
            num_of_samp_ticks = [4000, 6000, 8000, 10000, 12000, 14000]
            # ax_compression.set_xticks(num_of_samp_ticks, num_of_samp_ticks)
            # ax_compression.set_ylim([0.1, 1.4])

        # ### large scale 1 compression
        # if should_plot_compression(classification_scheme, compress_scheme, large_compression_idx=1):
        #     compression_rate = [result['compression'][classification_scheme][compress_scheme] for result in results.values()]
        #     ax_compression_of_large_scale_1.plot(samples_sizes, compression_rate, marker='o', ms=ms, label=label)
        # ##########
        #
        # ### large scale 2 compression
        # if should_plot_compression(classification_scheme, compress_scheme, large_compression_idx=2):
        #     compression_rate = [result['compression'][classification_scheme][compress_scheme] for result in results.values()]
        #     ax_compression_of_large_scale_2.plot(samples_sizes, compression_rate, marker='o', ms=ms, label=label)
        # ##########

        ### error ###
        error_probability = [result['error'][classification_scheme][compress_scheme] for result in results.values()]
        print(f'A: {classification_scheme} C: {compression_rate[0]} E: {error_probability[0]}')
        # ax_error.plot(samples_sizes, error_probability, marker='o', ms=ms, label=label)
        ax_error.plot(samples_sizes[0:], error_probability[0:], marker='o', ms=ms, label=label)
        # ax_error.set_xscale('log')
        # ax_error.set_yscale('log')
        ax_error.set_xlabel('Number of samples',fontsize = fontsize)
        ax_error.set_ylabel('Test error',fontsize = fontsize)
        # ax_error.set_yticks([.21,.22,.23,.24,.25,.26,.27,.28,.29],[.21,.22,.23,.24,.25,.26,.27,.28,.29])
        num_of_samp_ticks = [4000, 6000, 8000, 10000, 12000, 14000]
        # ax_error.set_xticks(num_of_samp_ticks, num_of_samp_ticks)
        ##########



    # loc = None
    loc = 'upper right'; bbox_to_anchor = (.9, .88)
    # fig_times.legend(loc=loc)
    # fig_compression.legend(loc=loc)
    fig_error.legend(loc=loc,bbox_to_anchor = bbox_to_anchor,fontsize = 11)

    out_path_dir = plots_params.out_path_dir
    if plots_params.bool_save_plots:
        if not out_path_dir.is_dir():
            out_path_dir.mkdir()
        # fig_times.savefig(out_path_dir / 'times.eps', format='eps')
        # fig_compression.savefig(out_path_dir / 'compression.eps', format='eps')
        # fig_error.savefig(out_path_dir / 'error.eps', format='eps')
        fig_times.savefig(out_path_dir / 'times.png')
        fig_compression.savefig(out_path_dir / 'compression.png')
        fig_error.savefig(out_path_dir / 'error.png')

    if plots_params.bool_show_plots:
        fig_times.show()
        fig_compression.show()
        # fig_compression_of_large_scale_1.show()
        # fig_compression_of_large_scale_2.show()
        fig_error.show()

    return


