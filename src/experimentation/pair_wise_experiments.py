import src.statistical_diffs.statistical_log_diff_analyzer as sld
from src.logs.log_generator import get_test_log
from src.statistical_modules.mle_for_log import *
from src.sampling.LogSampler import sample_traces
import itertools
import numpy as np


class Experiment_Result():

    def __init__(self, transitions_to_probabilities_per_log):
        '''
        :param transitions_to_probabilities_per_log: list of dictionaries of mapping k_futures -> k_futures for each of the logs
        '''
        self.T2Ps = transitions_to_probabilities_per_log
        self.results = None

    def add_experiment_result(self, k, min_diff, bias, alpha, found_diffs):

        new_row = [k, min_diff, bias, alpha]

        ## create a mapping of transitions to probabilities (in each of the logs)
        all_transitions = {}

        for T2P in self.T2Ps:
            ### go over the dict of each log
            for trans_pre in T2P:
                ### go over each equiv class
                trans_futures = T2P[trans_pre]
                for future_trans in trans_futures :
                    ### go over each future and add it to the mapping of transitions to probabilities
                    pr = trans_futures[future_trans]
                    transition = (trans_pre, future_trans)
                    transition_probabilities = all_transitions.get(transition, [])
                    transition_probabilities.append(pr)
                    all_transitions[transition] = transition_probabilities

        tp, tn, fp, fn = 0, 0, 0, 0
        for transition in all_transitions:
            transition_probabilities = all_transitions[transition]
            real_diff = abs(transition_probabilities[0] - transition_probabilities[1])
            if real_diff > bias:
                if transition in found_diffs:
                    tp += 1 ## predicted diff that exists
                else:
                    fn += 1 ## did not predict diff that exists
            else:
                if transition in found_diffs:
                    fp += 1 ## predicted diff that did not exists
                else:
                    tn += 1 ## did not predict diff that does not exists

        new_row.extend([tp, fp, tn, fn])
        self.results = np.vstack([self.results, new_row]) if self.results is not None else np.array(new_row)

    def __repr__(self):
        return str(self.results)

if __name__ == '__main__':


    ## Experiments main parameters
    ks = [2]
    min_diff = [0.0, 0.1, 0.2]
    biases = [0.0, 0.2, 0.4]
    alphas = [0.01, 0.05, 0.1]


    ## Repetition per configuration
    N = 200
    M = 10
    traces_to_sample = 1000
    for bias in biases:
        ## Logs
        log1 = get_test_log(0.0, 10000)
        log2 = get_test_log(bias, 10000)
        for (k, min_diff, alpha) in itertools.product(ks, min_diff, alphas):
            dict1 = compute_mle_k_future_dict(log1, k)
            dict2 = compute_mle_k_future_dict(log2, k)
            ## TODO: remove total_transitions
            ground_truth = [dict1, dict2]
            experiment_results = Experiment_Result(ground_truth)

            for trial in range(M): ## repeat the experiment for m randomally selected logs
                sampled_log1 = sample_traces(log1, traces_to_sample)
                sampled_log2 = sample_traces(log2, traces_to_sample)
                alg = sld.SLPDAnalyzer(sampled_log1, sampled_log2)
                diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                experiment_results.add_experiment_result(k, min_diff, bias, alpha, diffs)




