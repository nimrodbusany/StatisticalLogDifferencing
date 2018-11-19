from src.logs.LogFeatureExtractor import *


def compute_mle_k_future_dict(log, k):

    l = LogFeatureExtractor(log)
    return l.extract_k_sequences_to_future_dicts(k)


