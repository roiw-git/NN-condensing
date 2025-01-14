from pathlib import Path
from typing import Union


class PlotsParams:
    def __init__(
            self,
            bool_save_plots: bool,
            bool_show_plots: bool,
            out_path_dir: Union[str, Path]
    ):
        self.bool_save_plots = bool_save_plots
        self.bool_show_plots = bool_show_plots
        self.out_path_dir = Path(out_path_dir)
