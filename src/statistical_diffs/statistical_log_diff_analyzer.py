from src.statistical_modules.hypothesis_testing import proportions_comparison
from src.logs.log_generator import get_test_log

TOTAL_TRANSITIONS_ATTRIBUTE = 'total_transitions'
__VERBOSE__ = True

class SLPDAnalyzer:


    def __init__(self, logA, logB):
        self.logA = logA
        self.logB = logB

    def _extract_k_sequences_to_future_dicts(self, log, k):

        ksDict = {}
        for trace in log:
            for i in range(len(trace)):
                k_seq = trace[i : i + k + 1]
                future_k_seq = trace[i+1 : i + k + 2]
                futures = ksDict.get(k_seq, {})
                futures[future_k_seq] = futures.get(future_k_seq, 0) + 1
                ksDict[k_seq] = futures

        for k_seq in ksDict:
            futures = ksDict[k_seq]
            total_transitions = sum([x[1] for x in futures.items()])
            futures[TOTAL_TRANSITIONS_ATTRIBUTE] = total_transitions
            for future_k_seq in futures:
                futures[future_k_seq] = futures[future_k_seq] / float(total_transitions)
        return ksDict

    def find_diffs(self, k = 2, min_diff = 0.0, alpha=0.05):
        '''
        :param k: compare k-sequences of length
        :param min_diff: minimum difference to consider
        :param alpha: statistical bounds to consider
        :return: a set of
        '''
        dictA = self._extract_k_sequences_to_future_dicts(self.logA, k)
        dictB = self._extract_k_sequences_to_future_dicts(self.logB, k)

        k_seqs = set(dictA.keys())
        k_seqs.union(set(dictB.keys()))
        statistical_diffs = []
        for k_seq in k_seqs:
            future_a = dictA[k_seq]
            future_b = dictB[k_seq]
            future_k_seqs = set(future_a.keys())
            future_k_seqs.union(set(future_b.keys()))
            for future_seq in future_k_seqs:
                p1 = future_a[future_seq]
                n1 = future_a[TOTAL_TRANSITIONS_ATTRIBUTE]
                p2 = future_b[future_seq]
                n2 = future_b[TOTAL_TRANSITIONS_ATTRIBUTE]
                statistical_diff = proportions_comparison(p1, n1, p2, n2, min_diff, alpha)
                if __VERBOSE__:
                    if statistical_diff:
                        print('not diff found for:', k_seq, future_seq, '\np1, n1, p2, n2:', p1, n1, p2, n2)
                    else:
                        print('FOUND STATISTICAL DIFF:', k_seq, future_seq, '\np1, n1, p2, n2:', p1, n1, p2, n2)
                if statistical_diff:
                    statistical_diffs.append([k_seq, future_seq])

        return statistical_diffs


if __name__ == '__main__':

    min_diff = 0.0
    alpha = 0.05
    bias = 0.2
    k = 2
    log1 = get_test_log(0.0, 1000)
    log2 = get_test_log(bias, 1000)
    alg = SLPDAnalyzer(log1, log2)
    alg.find_diffs(k, min_diff, alpha)