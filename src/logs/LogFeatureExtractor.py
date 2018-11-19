
TOTAL_TRANSITIONS_ATTRIBUTE = 'total_transitions'

class LogFeatureExtractor():

    def __init__(self, log):
        self.log = log

    def extract_k_sequences_to_future_dicts(self, k): ### TODO: remove TOTAL_TRANSITIONS_ATTRIBUTE from dict, find
                                                        ### nicer solution

        ksDict = {}
        for trace in self.log:
            for i in range(len(trace)):
                k_seq = trace[i: i + k]
                future_k_seq = trace[i + 1: i + k + 1]
                futures = ksDict.get(k_seq, {})
                futures[future_k_seq] = futures.get(future_k_seq, 0) + 1
                ksDict[k_seq] = futures

        for k_seq in ksDict:
            futures = ksDict[k_seq]
            total_transitions = sum([x[1] for x in futures.items()])
            for future_k_seq in futures:
                futures[future_k_seq] = futures[future_k_seq] / float(total_transitions)
            futures[TOTAL_TRANSITIONS_ATTRIBUTE] = total_transitions
        return ksDict