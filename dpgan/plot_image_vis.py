import os
import sys
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver

import numpy as np
sys.path.insert(0, '../code_balanced/')

from aux import plot_mnist_batch


def first_look_dpgan():
  # mat = np.load('../dpgan/synth_data/apr17_sigma1.5_test/synthetic_mnist.npy')
  mat = np.load('synth_data/apr18_sigma1.5_baseline/synthetic_mnist.npz')
  # print(mat['data'])
  data = mat['data']
  targets = mat['labels']
  print(data.shape)
  print(targets.shape)
  print(np.max(data), np.min(data))

  data = np.reshape(data, (-1, 784))
  print(data.shape)

  mat1 = np.reshape(data[:100, :], (100, 784))
  plot_mnist_batch(mat1, 10, 10, save_path='dpgan_test_plot', denorm=False, save_raw=False)

  mat2 = data[6000:6100, :]
  plot_mnist_batch(mat2, 10, 10, save_path='dpgan_test_zeros', denorm=False, save_raw=False)


  mat3 = np.concatenate([data[targets[:, k] == 1.][:10] for k in range(10)])
  plot_mnist_batch(mat3, 10, 10, save_path='dpgan_test_classes', denorm=False, save_raw=False)


def vis_dpgan():
  os.makedirs('vis', exist_ok=True)
  for data_key in ('d', 'f'):
    for run in range(5):
      load_path = f'synth_data/apr19_sig1.41_{data_key}{run}/synthetic_mnist.npz'
      mat = np.load(load_path)
      # print(mat['data'])
      data = mat['data']
      targets = mat['labels']
      data = np.reshape(data, (-1, 784))
      mat3 = np.concatenate([data[targets[:, k] == 1.][:10] for k in range(10)])
      plot_mnist_batch(mat3, 10, 10, save_path=f'vis/plot_{data_key}{run}', denorm=False, save_raw=False)


if __name__ == '__main__':
  # first_look_dpgan()
  vis_dpgan()