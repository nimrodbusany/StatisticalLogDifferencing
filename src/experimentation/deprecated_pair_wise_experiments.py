import src.statistical_diffs.statistical_log_diff_analyzer as sld
from src.logs.log_generator import LogGeneator
from src.statistical_modules.mle_for_log import *
from src.sampling.LogSampler import sample_traces
import itertools
import numpy as np
import pandas as pd


class Experiment_Result():

    def __init__(self):
        self.results = None
        self.columns = ['l1_traces', 'l2_traces', 'k', 'min_diff', 'bias', 'alpha', 'tp', 'fp', 'tn', 'fn', 'precision',
                   'recall', 'acc', 'true_error (tp/tn+fp)', 'power (tp/tp+fn)']
        self.grp_by_columns = ['l1_traces', 'l2_traces', 'k', 'min_diff', 'bias', 'alpha']

    def add_experiment_result(self, T2Ps, k, min_diff, bias, alpha, statistical_diffs, M1, M2):
        '''
            :param transitions_to_probabilities_per_log: list of dictionaries of mapping k_futures -> k_futures for each of the logs
        '''
        new_row = [M1, M2, k, min_diff, bias, alpha]
        found_diffs = statistical_diffs
        ## create a mapping of transitions to probabilities (in each of the logs)
        all_transitions = {}

        for j in range(len(T2Ps)):
            T2P = T2Ps[j]
            ### go over the dict of each log
            for trans_pre in T2P:
                ### go over each equiv class
                trans_futures = T2P[trans_pre]
                for future_trans in trans_futures:
                    ### go over each future and add it to the mapping of transitions to probabilities
                    pr = trans_futures[future_trans]
                    transition = (trans_pre, future_trans)
                    transition_probabilities = all_transitions.get(transition, {})
                    transition_probabilities[j] = pr
                    all_transitions[transition] = transition_probabilities

        tp, tn, fp, fn = 0, 0, 0, 0
        for transition in all_transitions:
            transition_probabilities = all_transitions[transition]
            real_diff = abs(transition_probabilities.get(0, 0) - transition_probabilities.get(1, 0))
            significant_diff = False
            act_diff = found_diffs.get(transition)
            if act_diff is not None:
                significant_diff = act_diff['significant_diff']

            if real_diff > min_diff:
                if significant_diff:
                    tp += 1 ## predicted diff that exists
                else:
                    fn += 1 ## did not predict diff that exists
            else:
                if significant_diff:
                    fp += 1 ## predicted diff that did not exists
                else:
                    tn += 1 ## did not predict diff that does not exists

        precision = -1 if (tp+fp) == 0 else tp/(tp+fp)
        recall = -1 if (tp+fn) == 0 else tp / (tp+fn)
        statistical_error = -1 if (tn + fp) == 0 else fp / (tn + fp)
        power = -1 if (tp + fn) == 0 else tp / (tp + fn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        new_row.extend([tp, fp, tn, fn, precision, recall, acc, statistical_error, power])
        self.results = np.vstack([self.results, new_row]) if self.results is not None else np.array(new_row)

    def export_to_csv(self, path):
        with open(path, 'wb') as f:
            f.write(bytes(",".join(self.columns)+'\n','utf-8'))
            np.savetxt(f, self.results, delimiter=",", fmt='%.4f')

    def export_summary_to_csv(self, path):


        df = pd.DataFrame(data=self.results, columns=self.columns)
        bias = df['bias']
        df = df.replace(-1, np.NaN)
        df['bias'] = bias ## TODO: resolve this ugly hack; need to add bias, since group by with NaN values break the group by
        grps = df.groupby(self.grp_by_columns)
        res = grps.apply(np.mean)
        res.to_csv(path)

    def __repr__(self):
        return str(self.results)


def get_expreiment_logs(experiment_type= 0, bias= 0.1, full_log_size= 1000):

    if experiment_type == 0:
        log1 = LogGeneator.produce_log_from_single_split_models(0.0, full_log_size)
        log2 = LogGeneator.produce_log_from_single_split_models(bias, full_log_size)
        return log1, log2
    if experiment_type == 1:
        return LogGeneator.produce_toy_logs(bias, full_log_size)

    raise ValueError("experiment type: [0, 1]")


if __name__ == '__main__':

    RESULT_FODLER = '../../results/statistical_experiments/pair_wise/' ## statistical
    EXPERIMENT_TYPE = 1 # 0 to get logs from single split models, 1 to get toy logs with two k-sequences with a biase
    TYPE_TO_NAME = {0:'single_split', 1:'simple_toy'}
    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.011, 0.051, 0.11]
    biases = [0.01, 0.05, 0.1, 0.2, 0.4] ##  0.0, 0.2
    alphas = [0.01, 0.05, 0.1]


    ## Repetition per configuration
    M = 5
    full_log_size = 1000000
    traces_to_sample = [50, 500, 5000, 50000]
    experiment_results = Experiment_Result()
    for bias in biases:
        ## Logs
        log1, log2 = get_expreiment_logs(EXPERIMENT_TYPE, bias, full_log_size)
        for (k, min_diff, alpha) in itertools.product(ks, min_diffs, alphas):
            dict1 = compute_mle_k_future_dict(log1, k)
            dict2 = compute_mle_k_future_dict(log2, k)
            ## TODO: remove total_transitions
            ground_truth = [dict1, dict2]
            for sample in traces_to_sample:
                for trial in range(M): ## repeat the experiment for m randomally selected logs
                    sampled_log1 = sample_traces(log1, sample)
                    sampled_log2 = sample_traces(log2, sample)
                    alg = sld.SLPDAnalyzer(sampled_log1, sampled_log2)
                    diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                    experiment_results.add_experiment_result(ground_truth, k, min_diff, bias, alpha, diffs, sample, sample)


    exp_name = TYPE_TO_NAME[EXPERIMENT_TYPE]
    experiment_results.export_to_csv(
        RESULT_FODLER + exp_name + '_results' + '.csv')

    experiment_results.export_summary_to_csv(
        RESULT_FODLER + exp_name + '_results_summary' + '.csv')

                # experiment_results.export_to_csv(
    #     RESULT_FODLER + 'result_' + "_".join([str(k), str(min_diff), str(bias)]) + '.csv')



