from _constants import ALL, CompressionSchemes, ClassificationSchemes

user_params_dict = {
  "num_of_iterations": 1,
  "total_num_of_samples": 2310,  # ALL or int <= ~15K   [6K in the experiments]
  "num_of_samples_amounts": 1,  # number of samples sizes to plot (the "size" of x-axis)
  'training_percent': 0.7,  # float between 0 and 1
  "classification_schemes": [
    ClassificationSchemes.ONE_NN,
    # ClassificationSchemes.K_NN,
    # ClassificationSchemes.PROTO_K_NN,
    # ClassificationSchemes.OPTINET,
    ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC,
    # ClassificationSchemes.GREEDY_WEIGHTED_HEURISTIC_REDUCED,
    ClassificationSchemes.RSS,
    ClassificationSchemes.MSS
  ],  # [...], ALL, ClassificationSchemes
  "gamma": 0.11,
  "k": 10,
  "compression_schemes": [
    CompressionSchemes.NONE,
    # CompressionSchemes.PROTOCOMP,
    # CompressionSchemes.PROTOAPPROX,
  ],  # [...], None, ALL, CompressionSchemes
}