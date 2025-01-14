from enum import Enum, auto
from pathlib import Path

# SEED: int = 4358645
SEED: int = 4358691
# SEED: int = 4358689

ROOT_PATH = Path(__file__).parent

class ModelPhase(Enum):
    TRAIN = auto()
    TEST = auto()

class CompressionSchemes(Enum):
    NONE: str = 'none'
    PROTOCOMP: str = 'protocomp'
    PROTOAPPROX: str = 'protoapprox'


class ClassificationSchemes(Enum):
    ONE_NN = 'one_nn'
    K_NN = 'k_nn'
    PROTO_K_NN = 'proto_k_nn'
    OPTINET = 'optinet'
    GREEDY_WEIGHTED_HEURISTIC: str = 'greedy_weighted_heuristic'
    GREEDY_WEIGHTED_HEURISTIC_REDUCED: str = 'greedy_weighted_heuristic_reduced'
    RSS: str = 'rss'
    MSS: str = 'mss'

ALL = 'all'

legend_dict = {
    'one_nn' : '1-NN',
    'k_nn' : 'k-NN',
    'proto_k_nn' : 'Proto-k-NN',
    'optinet' : 'Optinet',
    'greedy_weighted_heuristic' : 'Greedy WNN',
    'rss' : 'RSS',
    'mss' : 'MSS',
    'protocomp' : 'ProtoComp',
}