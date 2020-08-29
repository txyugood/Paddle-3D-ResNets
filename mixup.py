import numpy as np
def create_mixup_reader(mixup_alpha, rd):
    """
    """

    class context:
        tmp_mix = []
        tmp_l1 = []
        tmp_l2 = []
        tmp_lam = []

    alpha = mixup_alpha

    def fetch_data():
        for item in rd():
            yield item

    def mixup_data():
        for data_list in fetch_data():
            if alpha > 0.:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.
            l1 = np.array(data_list)
            l2 = np.random.permutation(l1)
            mixed_l = [
                l1[i][0] * lam + (1 - lam) * l2[i][0] for i in range(len(l1))
            ]
            yield (mixed_l, l1, l2, lam)

    def mixup_reader():
        for context.tmp_mix, context.tmp_l1, context.tmp_l2, context.tmp_lam in mixup_data(
        ):
            for i in range(len(context.tmp_mix)):
                mixed_l = context.tmp_mix[i]
                l1 = context.tmp_l1[i]
                l2 = context.tmp_l2[i]
                lam = context.tmp_lam
                yield (mixed_l, l1[1], l2[1], float(lam))

    return mixup_reader