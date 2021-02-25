import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import FCCondGen, ConvCondGen
from aux import plot_mnist_batch, log_args, flatten_features, log_final_score
from data_loading import get_dataloaders
from rff_mmd_approx import get_rff_losses
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def
from synth_data_2d import plot_data


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
    data = flatten_features(data)

    loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      n_data = len(train_loader.dataset)
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))


def log_gen_data(gen, device, epoch, n_labels, log_dir):
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), n_labels)[:, None].to(device)
  gen_code, _ = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, n_labels, log_dir + f'samples_ep{epoch}', denorm=False)


def get_losses(ar, train_loader, device, n_feat, n_labels):
  # if ar.loss_type == 'real_mmd':
  #   minibatch_loss = get_real_mmd_loss(ar.rff_sigma, n_labels, ar.batch_size)
  #   single_release_loss = None
  # elif ar.loss_type == 'kmeans':
  #   minibatch_loss = None
  #   single_release_loss = get_kmeans_mmd_loss(train_loader, n_labels, ar.tgt_epsilon, ar.n_means,
  #                                             ar.rff_sigma, ar.batch_size, ar.dp_kmeans_encoding_dim)
  #
  # elif ar.loss_type == 'rff':
  single_release_loss, minibatch_loss, _ = get_rff_losses(train_loader, n_feat, ar.d_rff, ar.rff_sigma, device,
                                                          n_labels, ar.noise_factor, ar.mmd_type)
  # else:
  #   raise ValueError

  return minibatch_loss, single_release_loss


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits, fashion and 2d')
  parser.add_argument('--create-dataset', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=500)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--gen-batch-size', '-gbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  # parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--conv-gen', action='store_true', default=True, help='use convolutional generator')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='200', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=str, default=None, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=5.0, help='privacy noise parameter')

  # ALTERNATE MODES
  parser.add_argument('--single-release', action='store_true', default=True, help='get 1 data mean embedding only')

  parser.add_argument('--loss-type', type=str, default='rff', help='how to approx mmd',
                      choices=['rff', 'kmeans', 'real_mmd'])
  # parser.add_argument('--real-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')
  # parser.add_argument('--kmeans-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')

  parser.add_argument('--n-means', type=int, default=10, help='number of means to find per class')
  parser.add_argument('--dp-kmeans-encoding-dim', type=int, default=10, help='dimension the data is projected to')
  parser.add_argument('--tgt-epsilon', type=float, default=1.0, help='privacy epsilon for dp k-means')
  parser.add_argument('--kmeans-delta', type=float, default=0.01, help='soft failure probability in dp k-means')
  parser.add_argument('--mmd-type', type=str, default='sphere', help='how to approx mmd', choices=['sphere', 'r+r'])

  parser.add_argument('--center-data', action='store_true', default=False, help='k-means requires centering')

  # synth_d2 data
  parser.add_argument('--synth-spec-string', type=str, default='disc_k5_n10000_row5_col5_noise0.2', help='')
  parser.add_argument('--test-split', type=float, default=0.1, help='only relevant for synth_2d so far')

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
  assert ar.data in {'digits', 'fashion', '2d'}
  if ar.rff_sigma is None:
    ar.rff_sigma = '105' if ar.data == 'digits' else '127'

  if ar.loss_type == 'kmeans' and ar.tgt_epsilon > 0.0:
    assert ar.center_data, 'dp kmeans requires centering of data'

  if ar.data == '2d':
    ar.conv_gen = False
  else:
    ar.conv_gen = True


def synthesize_data_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  if n_data % gen_batch_size != 0:
    assert n_data % 100 == 0
    gen_batch_size = n_data // 100
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


def test_results(data_key, log_name, log_dir, data_tuple, eval_func):
  if data_key in {'digits', 'fashion'}:
    final_score = test_gen_data(log_name, data_key, subsample=0.1, custom_keys='logistic_reg')
    log_final_score(log_dir, final_score)
  elif data_key == '2d':
    final_score = test_passed_gen_data(log_name, data_tuple, log_save_dir=None, log_results=False,
                                       subsample=.1, custom_keys='mlp', compute_real_to_real=True)
    log_final_score(log_dir, final_score)
    eval_score = eval_func(data_tuple.x_gen, data_tuple.y_gen.flatten())
    print(f'Score of evaluation function: {eval_score}')
    with open(os.path.join(log_dir, 'eval_score'), 'w') as f:
      f.writelines([f'{eval_score}'])

    plot_data(data_tuple.x_real_train, data_tuple.y_real_train.flatten(), os.path.join(log_dir, 'plot_train'),
              center_frame=True)
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen'))
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_sub0.2'), subsample=0.2)
    plot_data(data_tuple.x_gen, data_tuple.y_gen.flatten(), os.path.join(log_dir, 'plot_gen_centered'),
              center_frame=True)


def main():
  # load settings
  ar = get_args()
  pt.manual_seed(ar.seed)
  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  # load data
  data_pkg = get_dataloaders(ar.data, ar.batch_size, ar.test_batch_size, use_cuda, ar.center_data,
                             ar.synth_spec_string, ar.test_split)

  # init model
  if ar.conv_gen:
    gen = ConvCondGen(ar.d_code, ar.gen_spec, data_pkg.n_labels, ar.n_channels, ar.kernel_sizes).to(device)
  else:
    use_sigmoid = ar.data in {'digits', 'fashion'}
    gen = FCCondGen(ar.d_code, ar.gen_spec, data_pkg.n_features, data_pkg.n_labels, use_sigmoid=use_sigmoid,
                    batch_norm=True).to(device)

  minibatch_loss, single_release_loss = get_losses(ar, data_pkg.train_loader, device,
                                                   data_pkg.n_features, data_pkg.n_labels)
  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  # training loop
  for epoch in range(1, ar.epochs + 1):
    if ar.single_release:
      train_single_release(gen, device, optimizer, epoch, single_release_loss, ar.log_interval, ar.batch_size,
                           data_pkg.n_data)
    else:
      train_multi_release(gen, device, data_pkg.train_loader, optimizer, epoch, minibatch_loss, ar.log_interval)

    # testing doesn't really inform how training is going, so it's commented out
    # test(gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size, ar.log_dir)
    if ar.data in {"digits", "fashion"}:
      log_gen_data(gen, device, epoch, data_pkg.n_labels, ar.log_dir)
    scheduler.step()

  # save trained model and data
  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')
  if ar.create_dataset:
    data_id = "synthetic_mnist" if ar.data in {"digits", "fashion"} else 'gen_data'

    # if ar.dataset in {'digits', 'fashion'}:
    syn_data, syn_labels = synthesize_data_with_uniform_labels(gen, device, gen_batch_size=ar.gen_batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)
    np.savez(ar.log_dir + data_id, data=syn_data, labels=syn_labels)

    data_tuple = datasets_colletion_def(syn_data, syn_labels,
                                        data_pkg.train_data.data, data_pkg.train_data.targets,
                                        data_pkg.test_data.data, data_pkg.test_data.targets)
    test_results(ar.data, ar.log_name, ar.log_dir, data_tuple, data_pkg.eval_func)
    # final_score = test_gen_data(ar.log_name, ar.data, subsample=0.1, custom_keys='logistic_reg')
    # log_final_score(ar.log_dir, final_score)


if __name__ == '__main__':
  main()
