import os
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import numpy as np


def make_dataset(n_classes, n_samples, n_rows, n_cols, noise_scale,
                 discrete=False, force_make_new=False, base_data_path='data/SYNTH2D/'):
  """
  @param n_classes:
  @param n_samples:
  @param n_rows:
  @param n_cols:
  @param noise_scale:
  @param discrete:
  @param force_make_new:
  @param base_data_path:

  @returns: the dataset, and a function to estimate the pdf
  """
  n_clusters = n_rows * n_cols
  assert n_clusters % n_classes == 0  # equal number of clusters per class
  assert n_samples % n_clusters == 0  # equal number  of samples per cluster
  assert (not discrete) or (noise_scale < 0.5)  # ensure no overlap in discrete case

  class_grid, center_grid, class_centers = create_centers(n_classes, n_clusters, n_rows, n_cols)
  print(class_grid)

  spec_str = specs_to_string(n_classes, n_samples, n_rows, n_cols, noise_scale, discrete)
  os.makedirs(os.path.join(base_data_path, spec_str), exist_ok=True)
  data_save_str = os.path.join(base_data_path, spec_str, 'samples.npz')
  if not force_make_new and os.path.exists(data_save_str):  # check if dataset already exists
    samples_mat = np.load(data_save_str)
    # print(samples_mat)
    data_samples = samples_mat['data']
    label_samples = samples_mat['labels']

  else:
    data_samples, label_samples = get_data_samples(center_grid, class_grid, n_rows, n_cols,
                                                   n_clusters, noise_scale, n_samples, discrete)
    np.savez(data_save_str, data=data_samples, labels=label_samples)

  if discrete:
    eval_func = get_discrete_in_out_test(class_centers, noise_scale)
  else:
    eval_func = get_mix_of_gaussian_pdf(class_centers, noise_scale)

  return data_samples, label_samples, eval_func, class_centers


def create_centers(n_classes, n_clusters, n_rows, n_cols):
  center_grid = np.stack([np.repeat(np.arange(n_rows)[:, None], n_cols, 1),
                          np.repeat(np.arange(n_cols)[None, :], n_rows, 0)], axis=2)

  # assign clusters to classes
  assert n_cols % n_classes == 0  # for now assume classes neatly fit into a row
  class_grid = np.zeros((n_rows, n_cols), dtype=np.int)
  class_centers = np.zeros((n_classes, n_clusters // n_classes, 2))
  centers_done_per_class = [0 for _ in range(n_classes)]
  next_class = 0
  for row_idx in range(n_rows):
    for col_idx in range(n_cols):
      class_grid[row_idx, col_idx] = next_class  # store class in grid
      class_centers[next_class, centers_done_per_class[next_class]] = center_grid[row_idx, col_idx]  # store center
      centers_done_per_class[next_class] += 1
      next_class = (next_class + 1) % n_classes

    next_class = (class_grid[row_idx, 0] + n_classes // 2) % n_classes
  return class_grid, center_grid, class_centers


def get_data_samples(center_grid, class_grid, n_rows, n_cols, n_clusters, noise_scale, n_samples, discrete):
  data_samples = []
  label_samples = []
  n_samples_per_cluster = n_samples // n_clusters
  for row_idx in range(n_rows):
    for col_idx in range(n_cols):
      data_samples.append(
        get_samples_from_center(center_grid[row_idx, col_idx], noise_scale, n_samples_per_cluster, discrete))
      label_samples.append(np.zeros((n_samples_per_cluster,), dtype=np.int) + class_grid[row_idx, col_idx])

  data_samples = np.concatenate(data_samples).astype(dtype=np.float32)
  label_samples = np.concatenate(label_samples)
  return data_samples, label_samples


def get_samples_from_center(center, noise_scale, n_samples, discrete):
  if discrete:
    square_sample = np.random.uniform(low=-noise_scale, high=noise_scale, size=(2*n_samples, 2))
    sample_norms = np.linalg.norm(square_sample, axis=1)
    sample = square_sample[sample_norms <= noise_scale][:n_samples]  # reject samples in corners
    sample += center
    if len(sample) < n_samples:
      print(len(sample), n_samples)
      more_samples = get_samples_from_center(center, noise_scale, n_samples - len(sample), discrete)
      sample = np.stack([sample, more_samples], axis=0)
  else:
    sample = np.random.normal(loc=center, scale=noise_scale, size=(n_samples, 2))
  return sample


def get_mix_of_gaussian_pdf(class_centers, noise_scale):
  print(class_centers.shape)
  n_classes, n_gaussians_per_class = class_centers.shape[:2]
  n_gaussians = n_classes * n_gaussians_per_class

  def mix_of_gaussian_pdf(data, labels):
    # evaluate each class separately
    data_prob_by_class = []
    for c_idx in range(n_classes):
      c_data = data[labels == c_idx]  # shape (n_c, 2)
      norms = np.linalg.norm(class_centers[c_idx].T[:, :, None] - c_data.T[:, None, :], axis=0)  # (c, n_c)
      # because we're using spherical gaussians, we can compute pdf as in 1D based on the norms
      gauss_probs = 1 / (np.sqrt(2 * np.pi) * noise_scale) * np.exp(-1/(2 * noise_scale**2) * norms**2) / n_gaussians
      prob_per_sample = np.sum(gauss_probs, axis=0)
      prob_per_sample = np.log(prob_per_sample)
      prob_c = np.sum(prob_per_sample)
      data_prob_by_class.append(prob_c)

    data_prob = np.sum(np.asarray(data_prob_by_class))
    return data_prob
  return mix_of_gaussian_pdf


def get_discrete_in_out_test(class_centers, noise_scale):
  n_classes, n_clusters_per_class = class_centers.shape[:2]

  def discrete_in_out_test(data, labels, return_per_class=False):
    # iterate through classes
    n_data_in = []
    n_data_out = []
    labels = labels.flatten()
    for c_idx in range(n_classes):
      print(data.shape, labels.shape, c_idx, (labels == c_idx).shape)
      c_data = data[labels == c_idx]  # shape (n_c, 2)
      # below: shape (2, c, 1) x (2, 1, n_c) -> (2, c, n_c) -> (c, n_c)
      norms = np.linalg.norm(class_centers[c_idx].T[:, :, None] - c_data.T[:, None, :], axis=0)
      # since each datapoint is within at most 1 centers' radius, we can sum the total number of times this is the case
      n_data_in_c = np.sum(norms <= noise_scale)
      assert n_data_in_c <= len(c_data)
      n_data_out_c = len(c_data) - n_data_in_c
      n_data_in.append(n_data_in_c)
      n_data_out.append(n_data_out_c)

    n_data_in = np.asarray(n_data_in)
    n_data_out = np.asarray(n_data_out)
    frac_in = np.sum(n_data_in) / len(data)

    if return_per_class:
      return frac_in, n_data_in, n_data_out
    else:
      return frac_in

  return discrete_in_out_test


def plot_data(data, labels, save_str, class_centers=None, subsample=None, center_frame=False, title=''):
  n_classes = int(np.max(labels)) + 1
  colors = ['r', 'b', 'g', 'y', 'orange', 'black', 'grey', 'cyan', 'magenta', 'brown']
  plt.figure()
  plt.title(title)
  if center_frame:
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)

  for c_idx in range(n_classes):
    c_data = data[labels == c_idx]

    if subsample is not None:
      n_sub = int(np.floor(len(c_data) * subsample))
      c_data = c_data[np.random.permutation(len(c_data))][:n_sub]

    plt.scatter(c_data[:, 1], c_data[:, 0], label=c_idx, c=colors[c_idx], s=.1)

    if class_centers is not None:
      print(class_centers[c_idx, 0, :])
      plt.scatter(class_centers[c_idx, :, 1], class_centers[c_idx, :, 0], marker='x', c=colors[c_idx], s=50.)

  plt.xlabel('x')
  plt.ylabel('y')
  # plt.legend()
  plt.savefig(f'{save_str}.png')


def specs_to_string(n_classes, n_samples, n_rows, n_cols, noise_scale, discrete):
  prefix = 'disc' if discrete else 'norm'
  return f'{prefix}_k{n_classes}_n{n_samples}_row{n_rows}_col{n_cols}_noise{noise_scale}'


def string_to_specs(spec_string):
  specs_list = spec_string.split('_')

  assert specs_list[0] in {'disc', 'norm'}
  assert specs_list[1][0] == 'k'
  assert specs_list[2][0] == 'n'
  assert specs_list[3][:3] == 'row'
  assert specs_list[4][:3] == 'col'
  assert specs_list[5][:5] == 'noise'

  discrete = specs_list[0] == 'disc'
  n_classes = int(specs_list[1][1:])
  n_samples = int(specs_list[2][1:])
  n_rows = int(specs_list[3][3:])
  n_cols = int(specs_list[4][3:])
  noise_scale = float(specs_list[5][5:])
  return n_classes, n_samples, n_rows, n_cols, noise_scale, discrete


def make_data_from_specstring(spec_string):
  n_classes, n_samples, n_rows, n_cols, noise_scale, discrete = string_to_specs(spec_string)
  data_samples, label_samples, eval_func, class_centers = make_dataset(n_classes, n_samples, n_rows, n_cols,
                                                                       noise_scale, discrete)
  return data_samples, label_samples, eval_func, class_centers


def main():
  data_samples, label_samples, eval_func, class_centers = make_dataset(n_classes=5,
                                                                       n_samples=100000,
                                                                       n_rows=5,
                                                                       n_cols=5,
                                                                       noise_scale=0.2,
                                                                       discrete=False)
  plot_data(data_samples, label_samples, 'synth_2d_data_plot', center_frame=True, title='')
  print(eval_func(data_samples, label_samples))


if __name__ == '__main__':
  main()
