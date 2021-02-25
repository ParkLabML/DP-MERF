# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
from autodp import rdp_acct, rdp_bank


def single_release_comp(sigma_1, sigma_2=None, delta=1e-5):
    """ input arguments """
    acct = rdp_acct.anaRDPacct()

    acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma_1}, x), prob=1.)
    if sigma_2 is not None:
        acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma_2}, x), prob=1.)

    print("Privacy loss is", acct.get_eps(delta))


if __name__ == '__main__':
    # main()
    # single sigma: sig=5 -> eps<1, sig=25 -> eps<0.2
    single_release_comp(25., sigma_2=None, delta=1e-5)
