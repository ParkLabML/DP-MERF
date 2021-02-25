# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
# by changing the form of upper bound on the Renyi DP, resulting from
# several Gaussian mechanisms we use given a mini-batch.
from autodp import rdp_acct, rdp_bank
import time
import numpy as np

def conservative_analysis():
  """ input arguments """

  # (1) privacy parameters for four types of Gaussian mechanisms
  sigma = 1.41

  # (2) desired delta level
  delta = 1e-5

  n_epochs = 10  # 5 for DP-MERF and 17 for DP-MERF+AE
  batch_size = 64  # the same across experiments
  acct = rdp_acct.anaRDPacct()

  n_data_by_class = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

  start_time = time.time()
  for n_data in n_data_by_class:

    steps_per_epoch = int(np.ceil(n_data / batch_size))
    n_steps = steps_per_epoch * n_epochs
    sampling_rate = batch_size / n_data

    epoch_last_batch_size = n_data % batch_size
    epoch_last_sampling_rate = epoch_last_batch_size / n_data

    subset_count = 0
    old_time = start_time
    for i in range(1, n_steps + 1):
      sampling_rate_i = epoch_last_sampling_rate if i % steps_per_epoch == 0 else sampling_rate
      acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), sampling_rate_i)
      if i % steps_per_epoch == 0:
        new_time = time.time()
        epochs_done = i // steps_per_epoch
        print(f'Epoch {epochs_done} done - Time used: {new_time - old_time}, Total: {new_time - start_time}')
        old_time = new_time

      if i == n_steps:
        pre_eps_time = time.time()
        subset_count += 1
        print("[", i, "]Privacy loss is", (acct.get_eps(delta)))
        post_eps_time = time.time()
        print('time to get_eps: ', post_eps_time - pre_eps_time)
        old_time = post_eps_time
    print(f'data subset {subset_count} done')


def main():
    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma = 1.2

    # (2) desired delta level
    delta = 1e-5

    # (5) number of training steps
    n_epochs = 10# 5 for DP-MERF and 17 for DP-MERF+AE
    batch_size = 64  # the same across experiments

    dataset="intrusion"

    if dataset == "epileptic":
        n_data = 8049
    elif dataset == "isolet":
        n_data = 4366
    elif dataset == "adult":
        n_data = 11077
    elif dataset == "census":
        n_data = 199523
    elif dataset == "cervical":
        n_data = 753
    elif dataset == "credit":
        n_data = 2668
    elif dataset == "intrusion":
        n_data = 394021
    elif dataset == "covtype":
        n_data = 9217

    steps_per_epoch = n_data // batch_size
    n_steps = steps_per_epoch * n_epochs
    # n_steps = 1

    # (6) sampling rate
    prob = batch_size / n_data
    # prob = 1

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    eps_seq = []

    for i in range(1, n_steps+1):
        acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
        if i % steps_per_epoch == 0 or i == n_steps:
            eps_seq.append(acct.get_eps(delta))
            print("[", i, "]Privacy loss is", (eps_seq[-1]))


if __name__ == '__main__':
    main()
    #conservative_analysis()