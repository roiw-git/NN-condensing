from pathlib import Path
from typing import Union

class ProgramParams:
    def __init__(
            self,
            run_new_experiment: bool,
            show_in_run_stats: bool,
            show_profile_stats: bool,
            compute_all_distances_first: bool,
            load_all_distances: bool,
            out_path_dir: Union[str, Path],

    ):
        self.run_new_experiment = run_new_experiment
        self.show_in_run_stats = show_in_run_stats
        self.show_profile_stats = show_profile_stats
        self.store_results = True
        self.compute_all_distances_first = compute_all_distances_first
        self.load_all_distances = load_all_distances
        self.out_path_dir = Path(out_path_dir)
