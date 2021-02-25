import os
import numpy as np
import torch as pt
from collections import namedtuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from aux import flip_mnist_data
from synth_data_2d import make_data_from_specstring, string_to_specs


train_data_tuple_def = namedtuple('train_data_tuple', ['train_loader', 'test_loader',
                                                       'train_data', 'test_data',
                                                       'n_features', 'n_data', 'n_labels', 'eval_func'])


def get_dataloaders(dataset_key, batch_size, test_batch_size, use_cuda, normalize, synth_spec_string, test_split):
  if dataset_key in {'digits', 'fashion'}:
    train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(batch_size, test_batch_size, use_cuda,
                                                                          dataset=dataset_key, normalize=normalize,
                                                                          return_datasets=True)
    n_features = 784
    n_data = 60_000
    n_labels = 10
    eval_func = None
  elif dataset_key == '2d':
    train_loader, trn_data, tst_data, eval_func = get_2d_synth_dataloaders(batch_size, use_cuda, synth_spec_string,
                                                                           normalize, test_split)
    test_loader = None
    n_labels, n_data, _, _, _, _ = string_to_specs(synth_spec_string)
    n_features = 2
  else:
    raise ValueError

  return train_data_tuple_def(train_loader, test_loader, trn_data, tst_data, n_features, n_data, n_labels, eval_func)


def get_mnist_dataloaders(batch_size, test_batch_size, use_cuda, normalize=False,
                          dataset='digits', data_dir='data', flip=False, return_datasets=False):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  transforms_list = [transforms.ToTensor()]
  if dataset == 'digits':
    if normalize:
      mnist_mean = 0.1307
      mnist_sdev = 0.3081
      transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.MNIST(data_dir, train=False, transform=prep_transforms)
    if flip:
      assert not normalize
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data)
      flip_mnist_data(tst_data)

    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  elif dataset == 'fashion':
    assert not normalize
    prep_transforms = transforms.Compose(transforms_list)
    trn_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=prep_transforms)
    tst_data = datasets.FashionMNIST(data_dir, train=False, transform=prep_transforms)
    if flip:
      print(pt.max(trn_data.data))
      flip_mnist_data(trn_data)
      flip_mnist_data(tst_data)
    train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
  else:
    raise ValueError

  if return_datasets:
    return train_loader, test_loader, trn_data, tst_data
  else:
    return train_loader, test_loader


class Synth2DDataset(Dataset):
  def __init__(self, data, targets):

    self.data = data
    self.targets = targets

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.targets[idx]


def get_2d_synth_dataloaders(batch_size, use_cuda, spec_string, normalize=False, test_split=None):
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  # transforms_list = [transforms.ToTensor()]
  # prep_transforms = transforms.Compose(transforms_list)

  assert not normalize  # maybe later
  data_samples, label_samples, eval_func, class_centers = make_data_from_specstring(spec_string)

  if test_split is not None:  #
    trn_data = []
    trn_labels = []
    tst_data = []
    tst_labels = []

    for label in range(np.max(label_samples)+1):
      sub_data = data_samples[label_samples == label]
      n_sub = len(sub_data)
      sub_data = sub_data[np.random.permutation(n_sub)]
      n_sub_tst = int(np.floor(n_sub * test_split))
      tst_data.append(sub_data[:n_sub_tst])
      trn_data.append(sub_data[n_sub_tst:])
      tst_labels.append(np.zeros(n_sub_tst, dtype=np.int) + label)
      trn_labels.append(np.zeros(n_sub - n_sub_tst, dtype=np.int) + label)

    data_samples = np.concatenate(trn_data)
    label_samples = np.concatenate(trn_labels)

    tst_data = np.concatenate(tst_data)
    tst_labels = np.concatenate(tst_labels)
    tst_data = Synth2DDataset(tst_data, tst_labels)
  else:
    tst_data = Synth2DDataset(None, None)

  trn_data = Synth2DDataset(data_samples, label_samples)

  train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
  return train_loader, trn_data, tst_data, eval_func