import numpy as np
from scipy import stats

__VERBOSE__ = False


def means_comparison(N = 1000):

    for i in range(N):
        count_success = 0
        n1 = 1000
        n2 = 1000
        hypo_diff = 0.00
        actual_diff = 0.1
        a = np.random.choice(2, p=[0.5 - actual_diff, 0.5 + actual_diff], size = n1)
        b = np.random.choice(2, p=[0.5, 0.5], size = n2)
        # res = sp.statistical_modules.ttest_ind(a, b, equal_var=False, )
        var_1 = a.var(ddof=1)
        var_2 = b.var(ddof=1)
        se = np.sqrt( var_1/n1 + var_2/n2 )
        df = (var_1/n1 + var_2/n2)**2 / (( (var_1 / n1)**2/(n1 - 1)) + ((var_2 / n2)**2 / (n2 - 1)))
        t = (a.mean() - b.mean() - hypo_diff) / se
        p = 1 - stats.t.cdf(t, df=df)
        if p < 0.05:
            count_success += 1
        if __VERBOSE__:
            print('t, df:', t, df)
            print('p_value is:', p)

    print ('success rate:', count_success / N)


def proportions_comparison(p1, n1, p2, n2, delta, alpha):

    p_hat = (round(p1 * n1) + round(p2 * n2)) / (n1 + n2)
    se = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
    z_hat = (abs(p1 - p2) - delta) / se
    p = 1 - stats.norm.cdf(z_hat)
    if p < alpha:
        return True
    return False

def proportions_comparison_test(N = 1000):

    def compute_cohens_h(a, n1, b, n2):
        return abs(2 * np.arcsin( (sum(a) / n1) ** 0.5) - 2 * np.arcsin( (sum(b) / n2) ** 0.5))

    count_success = 0
    cohens_h_arr = []
    for i in range(N):
        n1 = 1000
        n2 = 1000
        actual_diff = 0.45
        a = np.random.choice(2, p=[0.5 - actual_diff, 0.5 + actual_diff], size = n1)
        b = np.random.choice(2, p=[0.5, 0.5], size = n2)
        p_hat = (sum(a) + sum(b)) / (n1 + n2)
        se = np.sqrt(p_hat * (1-p_hat) * (1/n1 + 1/n2))
        z_hat = (a.mean() - b.mean() - actual_diff) / se
        p = 1 - stats.norm.cdf(z_hat)
        if p < 0.05:
            count_success += 1
        cohens_h_arr.append(compute_cohens_h(a, n1, b, n2))
    print('avg. success rate:', count_success / N)
    print('avg. cohens\' rate:', np.mean(cohens_h_arr))
