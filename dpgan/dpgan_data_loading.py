import os
import torch as pt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import sys

sys.path.append("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR")
# from data.tab_dataloader import load_credit, load_isolet, load_epileptic, load_adult, load_cervical, load_census, load_intrusion, load_covtype
try:
    from synth_data_2d import make_data_from_specstring, string_to_specs, plot_data
except ImportError:
    print('importing through relative path')
    # Import required Differential Privacy packages
    baseDir = "../code/code_balanced/"
    sys.path.append(baseDir)

    from synth_data_2d import make_data_from_specstring, string_to_specs, plot_data


def get_dataloader(batch_size):
  # Configure data loader
  os.makedirs("../data", exist_ok=True)
  data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  dataset = datasets.MNIST("../data", train=True, download=True, transform=data_transform)
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader


def get_single_label_dataloader(batch_size, label_idx, data_key='digits'):
  os.makedirs("../data", exist_ok=True)
  data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

  assert data_key in {'digits', 'fashion'}
  if data_key == 'digits':
    dataset = datasets.MNIST("../data", train=True, download=True, transform=data_transform)
  else:
    dataset = datasets.FashionMNIST("../data", train=True, download=True, transform=data_transform)
  selected_ids = dataset.targets == label_idx
  dataset.data = dataset.data[selected_ids]
  dataset.targets = dataset.targets[selected_ids]
  n_data = dataset.data.shape[0]
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader, n_data


class Synth2DDataset(Dataset):
  def __init__(self, data, labels):

    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


def get_single_label_dataloader_syn2d(batch_size, label_idx, data_spec_string):
  # os.makedirs("../data", exist_ok=True)
  # data_transform = transforms.Compose([transforms.ToTensor()])

  # assert data_key in {'digits', 'fashion'}
  data_samples, label_samples, eval_func, class_centers = make_data_from_specstring(data_spec_string)
  dataset = Synth2DDataset(data_samples, label_samples)
  # dataset = datasets.MNIST("../data", train=True, download=True, transform=data_transform)

  selected_ids = dataset.labels == label_idx
  dataset.data = dataset.data[selected_ids]
  dataset.targets = dataset.labels[selected_ids]
  n_data = dataset.data.shape[0]
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader, n_data



def get_single_label_dataloader_tab(batch_size, label_idx, data_key='isolet'):
  sys.path.append("/home/anon_k/Desktop/Dropbox/Current_research/privacy/DPDR")
  from data.tab_dataloader import load_credit, load_isolet, load_epileptic, load_adult, load_cervical, load_census, \
    load_intrusion, load_covtype

  [load_isolet, load_credit, load_cervical, load_epileptic, load_adult, load_intrusion, load_covtype]

  X_train, y_train, X_test, y_test = globals()["load_"+data_key]()
  print(f"Shape of the training data is {X_train.shape}")

  tensor_x=pt.stack([pt.Tensor(i) for i in X_train])
  tensor_y=pt.stack([pt.Tensor(np.array([i])) for i in y_train])
  #dataset = TensorDataset(tensor_x, targets)
  #dataloader = DataLoader(dataset)

  selected_ids = tensor_y == label_idx
  print(f"{label_idx} has {sum(selected_ids.squeeze())} data samples")
  tensor_x_selected = tensor_x[selected_ids.squeeze()]
  tensor_y_selected = tensor_y[selected_ids.squeeze()]
  n_data = len(tensor_y_selected)
  dataset = TensorDataset(tensor_x_selected, tensor_y_selected)
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader, n_data, X_test, y_test



if __name__ == '__main__':
  for i in range(102035):
    get_single_label_dataloader_tab(100, i, "covtype")

