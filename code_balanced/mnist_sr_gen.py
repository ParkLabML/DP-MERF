import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import ConvCondGen
from aux import plot_mnist_batch, meddistance, log_args, flat_data, log_final_score
from data_loading import get_mnist_dataloaders
from rff_mmd_approx import get_rff_losses
from synth_data_benchmark import test_gen_data


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data):
  n_iter = n_data // batch_size
  for batch_idx in range(n_iter):
    gen_code, gen_labels = gen.get_code(batch_size, device)
    loss = rff_mmd_loss(gen(gen_code), gen_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device):
  bs = labels.shape[0]
  gen_code, gen_labels = gen.get_code(bs, device)
  gen_samples = gen(gen_code)
  return rff_mmd_loss(data, labels, gen_samples, gen_labels)


def train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval):

  for batch_idx, (data, labels) in enumerate(train_loader):
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      n_data = len(train_loader.dataset)
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, log_dir):
  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data, labels = data.to(device), labels.to(device)
      data = flat_data(data, labels, device, n_labels=10, add_label=False)
      loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
      test_loss += loss.item()  # sum up batch loss

  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
  gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}', denorm=False)
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--n-labels', type=int, default=10, help='number of labels/classes in data')
  parser.add_argument('--log-interval', type=int, default=10000, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--synth-code_balanced', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=100)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  # parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='200', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')
  parser.add_argument('--mmd-type', type=str, default='sphere', help='how to approx mmd', choices=['sphere', 'r+r'])

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=str, default=None, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=5.0, help='privacy noise parameter')

  parser.add_argument('--flip-mnist', action='store_true', default=False, help='')

  ar = parser.parse_args()

  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def preprocess_args(ar):
  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)

  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)
  assert ar.data in {'digits', 'fashion'}
  if ar.rff_sigma is None:
    ar.rff_sigma = '105' if ar.data == 'digits' else '127'


def synthesize_mnist_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  assert n_data % gen_batch_size == 0
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def main():
  # load settings
  n_data, n_feat = 60000, 784

  ar = get_args()
  pt.manual_seed(ar.seed)
  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  # load data
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda, dataset=ar.data,
                                                    flip=ar.flip_mnist)

  # init model
  gen = ConvCondGen(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes).to(device)

  # define loss function

  sr_loss, mb_loss, _ = get_rff_losses(train_loader, n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor,
                                       ar.mmd_type)

  # rff_mmd_loss = get_rff_mmd_loss(n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  # training loop
  for epoch in range(1, ar.epochs + 1):
    train_single_release(gen, device, optimizer, epoch, sr_loss, ar.log_interval, ar.batch_size, n_data)
    test(gen, device, test_loader, mb_loss, epoch, ar.batch_size, ar.log_dir)
    scheduler.step()

  # save trained model and data
  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')

  syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, device)
  np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)

  test_model_key = 'mlp' if ar.flip_mnist else 'logistic_reg'
  final_score = test_gen_data(ar.log_name, ar.data, subsample=0.1, custom_keys=test_model_key, data_from_torch=True)
  log_final_score(ar.log_dir, final_score)


if __name__ == '__main__':
  main()
