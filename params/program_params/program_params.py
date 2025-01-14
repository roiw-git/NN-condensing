from _constants import ROOT_PATH

program_params_dict = {
    "run_new_experiment":  True,  # if False, it takes the stats of the last experimet and plot it (if it is asked to plot)
    "show_in_run_stats": True,
    "show_profile_stats": False,
    "compute_all_distances_first": False,
    "load_all_distances": False,
    'out_path_dir': ROOT_PATH / 'resources' / 'out' / 'magic',
}