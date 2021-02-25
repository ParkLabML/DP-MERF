import os
from collections import defaultdict, namedtuple
import numpy as np
import argparse
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost
import time


def load_mnist_data(data_key, data_from_torch, base_dir='data/'):
  if not data_from_torch:
    if data_key == 'digits':
      d = np.load(os.path.join(base_dir, 'MNIST/numpy_dmnist.npz'))  # x_train=x_trn, y_train=y_trn, x_test=x_tst, y_test=y_tst
      return d['x_train'], d['y_train'], d['x_test'], d['y_test']
    elif data_key == 'fashion':
      d = np.load(os.path.join(base_dir, 'FashionMNIST/numpy_fmnist.npz'))
      return d['x_train'], d['y_train'], d['x_test'], d['y_test']
    else:
      raise ValueError
  else:
    from torchvision import datasets
    if data_key == 'digits':
      train_data = datasets.MNIST('data', train=True)
      test_data = datasets.MNIST('data', train=False)
    elif data_key == 'fashion':
      train_data = datasets.FashionMNIST('data', train=True)
      test_data = datasets.FashionMNIST('data', train=False)
    else:
      raise ValueError

    x_real_train, y_real_train = train_data.data.numpy(), train_data.targets.numpy()
    x_real_train = np.reshape(x_real_train, (-1, 784)) / 255

    x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
    x_real_test = np.reshape(x_real_test, (-1, 784)) / 255
    return x_real_train, y_real_train, x_real_test, y_real_test


def subsample_data(x, y, frac, balance_classes=True):
  n_data = y.shape[0]
  n_classes = np.max(y) + 1
  new_n_data = int(n_data * frac)
  if not balance_classes:
    x, y = x[:new_n_data], y[:new_n_data]
  else:
    n_data_per_class = new_n_data // n_classes
    assert n_data_per_class * n_classes == new_n_data
    # print(f'starting label count {[sum(y == k) for k in range(n_classes)]}')
    # print('DEBUG: NCLASSES', n_classes, 'NDATA', n_data)
    rand_perm = np.random.permutation(n_data)
    x = x[rand_perm]
    y = y[rand_perm]
    # y_scalar = np.argmax(y, axis=1)

    data_ids = [[], [], [], [], [], [], [], [], [], []]
    n_full = 0
    for idx in range(n_data):
      l = y[idx]
      if len(data_ids[l]) < n_data_per_class:
        data_ids[l].append(idx)
        # print(l)
        if len(data_ids[l]) == n_data_per_class:
          n_full += 1
          if n_full == n_classes:
            break

    data_ids = np.asarray(data_ids)
    data_ids = np.reshape(data_ids, (new_n_data,))
    rand_perm = np.random.permutation(new_n_data)
    data_ids = data_ids[rand_perm]  # otherwise sorted by class
    x = x[data_ids]
    y = y[data_ids]

    print(f'subsampled label count {[sum(y == k) for k in range(n_classes)]}')
  return x, y


def normalize_data(x_train, x_test):
  mean = np.mean(x_train)
  sdev = np.std(x_train)
  x_train_normed = (x_train - mean) / sdev
  x_test_normed = (x_test - mean) / sdev
  assert not np.any(np.isnan(x_train_normed)) and not np.any(np.isnan(x_test_normed))

  return x_train_normed, x_test_normed


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-path', type=str, default=None, help='this is computed. only set to override')
  parser.add_argument('--data-base-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--data-log-name', type=str, default=None, help='subdirectory for this run')

  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--shuffle-data', action='store_true', default=False, help='shuffle data before testing')

  parser.add_argument('--log-results', action='store_true', default=False, help='if true, save results')
  parser.add_argument('--print-conf-mat', action='store_true', default=False, help='print confusion matrix')

  parser.add_argument('--skip-slow-models', action='store_true', default=False, help='skip models that take longer')
  parser.add_argument('--only-slow-models', action='store_true', default=False, help='only do slower the models')
  parser.add_argument('--custom-keys', type=str, default=None, help='enter model keys to run as key1,key2,key3...')

  parser.add_argument('--skip-gen-to-real', action='store_true', default=False, help='skip train:gen,test:real setting')
  parser.add_argument('--compute-real-to-real', action='store_true', default=False, help='add train:real,test:real')
  parser.add_argument('--compute-real-to-gen', action='store_true', default=False, help='add train:real,test:gen')

  parser.add_argument('--subsample', type=float, default=1., help='fraction on data to use in training')
  parser.add_argument('--sub-balanced-labels', action='store_true', default=False, help='add train:real,test:gen')

  parser.add_argument('--data-from-torch', action='store_true', default=True, help='if true, load data from pytorch')

  parser.add_argument('--norm-data', action='store_true', default=False, help='if true, normalize data (mostly debug)')

  ar = parser.parse_args()
  return ar


datasets_colletion_def = namedtuple('datasets_collection', ['x_gen', 'y_gen',
                                                            'x_real_train', 'y_real_train',
                                                            'x_real_test', 'y_real_test'])

def prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels):
  x_real_train, y_real_train, x_real_test, y_real_test = load_mnist_data(data_key, data_from_torch)
  gen_data = np.load(data_path)
  x_gen, y_gen = gen_data['data'], gen_data['labels']
  if len(y_gen.shape) == 2:  # remove onehot
    if y_gen.shape[1] == 1:
      y_gen = y_gen.ravel()
    elif y_gen.shape[1] == 10:
      y_gen = np.argmax(y_gen, axis=1)
    else:
      raise ValueError

  if shuffle_data:
    rand_perm = np.random.permutation(y_gen.shape[0])
    x_gen, y_gen = x_gen[rand_perm], y_gen[rand_perm]

  if subsample < 1.:
    x_gen, y_gen = subsample_data(x_gen, y_gen, subsample, sub_balanced_labels)
    x_real_train, y_real_train = subsample_data(x_real_train, y_real_train, subsample, sub_balanced_labels)

    print(f'training on {subsample * 100.}% of the original syntetic dataset')

  print(f'data ranges: [{np.min(x_real_test)}, {np.max(x_real_test)}], [{np.min(x_real_train)}, '
        f'{np.max(x_real_train)}], [{np.min(x_gen)}, {np.max(x_gen)}]')
  print(f'label ranges: [{np.min(y_real_test)}, {np.max(y_real_test)}], [{np.min(y_real_train)}, '
        f'{np.max(y_real_train)}], [{np.min(y_gen)}, {np.max(y_gen)}]')

  return datasets_colletion_def(x_gen, y_gen, x_real_train, y_real_train, x_real_test, y_real_test)


def prep_models(custom_keys, skip_slow_models, only_slow_models):
  assert not (skip_slow_models and only_slow_models)

  models = {'logistic_reg': linear_model.LogisticRegression,
            'random_forest': ensemble.RandomForestClassifier,
            'gaussian_nb': naive_bayes.GaussianNB,
            'bernoulli_nb': naive_bayes.BernoulliNB,
            'linear_svc': svm.LinearSVC,
            'decision_tree': tree.DecisionTreeClassifier,
            'lda': discriminant_analysis.LinearDiscriminantAnalysis,
            'adaboost': ensemble.AdaBoostClassifier,
            'mlp': neural_network.MLPClassifier,
            'bagging': ensemble.BaggingClassifier,
            'gbm': ensemble.GradientBoostingClassifier,
            'xgboost': xgboost.XGBClassifier}

  slow_models = {'bagging', 'gbm', 'xgboost'}

  model_specs = defaultdict(dict)
  model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
  model_specs['random_forest'] = {'n_estimators': 100, 'class_weight': 'balanced'}
  model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
  model_specs['bernoulli_nb'] = {'binarize': 0.5}
  model_specs['lda'] = {'solver': 'eigen', 'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
  model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                  'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                  'min_impurity_decrease': 0.0}
  model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}  # setting used in neurips2020 submission
  # model_specs['adaboost'] = {'n_estimators': 100, 'learning_rate': 0.1, 'algorithm': 'SAMME.R'}  best so far
  model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
  model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
  model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

  if custom_keys is not None:
    run_keys = custom_keys.split(',')
  elif skip_slow_models:
    run_keys = [k for k in models.keys() if k not in slow_models]
  elif only_slow_models:
    run_keys = [k for k in models.keys() if k in slow_models]
  else:
    run_keys = models.keys()

  return models, model_specs, run_keys


def model_test_run(model, x_tr, y_tr, x_ts, y_ts, norm_data, acc_str, f1_str):
  x_tr, x_ts = normalize_data(x_tr, x_ts) if norm_data else (x_tr, x_ts)
  model.fit(x_tr, y_tr)
  y_pred = model.predict(x_ts)
  acc = accuracy_score(y_pred, y_ts)
  f1 = f1_score(y_true=y_ts, y_pred=y_pred, average='macro')
  conf = confusion_matrix(y_true=y_ts, y_pred=y_pred)
  acc_str = acc_str + f' {acc}'
  f1_str = f1_str + f' {f1}'
  return acc, f1, conf, acc_str, f1_str


def test_gen_data(data_log_name, data_key, data_base_dir='logs/gen/', log_results=False, data_path=None,
                  data_from_torch=False, shuffle_data=False, subsample=1., sub_balanced_labels=True,
                  custom_keys=None, skip_slow_models=False, only_slow_models=False,
                  skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                  print_conf_mat=False, norm_data=False):

  gen_data_dir = os.path.join(data_base_dir, data_log_name)
  log_save_dir = os.path.join(gen_data_dir, 'synth_eval/')
  if data_path is None:
    data_path = os.path.join(gen_data_dir, 'synthetic_mnist.npz')
  datasets_colletion = prep_data(data_key, data_from_torch, data_path, shuffle_data, subsample, sub_balanced_labels)
  mean_acc = test_passed_gen_data(data_log_name, datasets_colletion, log_save_dir, log_results,
                                  subsample, custom_keys, skip_slow_models, only_slow_models,
                                  skip_gen_to_real, compute_real_to_real, compute_real_to_gen,
                                  print_conf_mat, norm_data)
  return mean_acc


def test_passed_gen_data(data_log_name, datasets_colletion, log_save_dir, log_results=False,
                         subsample=1., custom_keys=None, skip_slow_models=False, only_slow_models=False,
                         skip_gen_to_real=False, compute_real_to_real=False, compute_real_to_gen=False,
                         print_conf_mat=False, norm_data=False):
  if data_log_name is not None:
    print(f'processing {data_log_name}')


  if log_results:
    os.makedirs(log_save_dir, exist_ok=True)


  models, model_specs, run_keys = prep_models(custom_keys, skip_slow_models, only_slow_models)

  g_to_r_acc_summary = []
  dc = datasets_colletion
  for key in run_keys:
    print(f'Model: {key}')
    a_str, f_str = 'acc:', 'f1:'

    if not skip_gen_to_real:
      model = models[key](**model_specs[key])
      g_to_r_acc, g_to_r_f1, g_to_r_conf, a_str, f_str = model_test_run(model, dc.x_gen, dc.y_gen,
                                                                        dc.x_real_test, dc.y_real_test,
                                                                        norm_data, a_str + 'g2r', f_str + 'g2r')
      g_to_r_acc_summary.append(g_to_r_acc)
    else:
      g_to_r_acc, g_to_r_f1, g_to_r_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_real:
      model = models[key](**model_specs[key])
      base_acc, base_f1, base_conf, a_str, f_str = model_test_run(model,
                                                                  dc.x_real_train, dc.y_real_train,
                                                                  dc.x_real_test, dc.y_real_test,
                                                                  norm_data, a_str + 'r2r', f_str + 'r2r')
    else:
      base_acc, base_f1, base_conf = -1, -1, -np.ones((10, 10))

    if compute_real_to_gen:
      model = models[key](**model_specs[key])
      r_to_g_acc, r_to_g_f1, r_to_g_conv, a_str, f_str = model_test_run(model,
                                                                        dc.x_real_train, dc.y_real_train,
                                                                        dc.x_gen[:10000], dc.y_gen[:10000],
                                                                        norm_data, a_str + 'r2g', f_str + 'r2g')
    else:
      r_to_g_acc, r_to_g_f1, r_to_g_conv = -1, -1, -np.ones((10, 10))

    print(a_str)
    print(f_str)
    if print_conf_mat:
      print('gen to real confusion matrix:')
      print(g_to_r_conf)

    if log_results:
      accs = np.asarray([base_acc, g_to_r_acc, r_to_g_acc])
      f1_scores = np.asarray([base_f1, g_to_r_f1, r_to_g_f1])
      conf_mats = np.stack([base_conf, g_to_r_conf, r_to_g_conv])
      file_name = f'sub{subsample}_{key}_log'
      np.savez(os.path.join(log_save_dir, file_name), accuracies=accs, f1_scores=f1_scores, conf_mats=conf_mats)

  print('acc summary:')
  for acc in g_to_r_acc_summary:
    print(acc)
  mean_acc = np.mean(g_to_r_acc_summary)
  print(f'mean: {mean_acc}')
  return mean_acc


def main():
  ar = parse()
  test_gen_data(ar.data_log_name, ar.data, ar.data_base_dir, ar.log_results, ar.data_path, ar.data_from_torch,
                ar.shuffle_data, ar.subsample, ar.sub_balanced_labels, ar.custom_keys,
                ar.skip_slow_models, ar.only_slow_models,
                ar.skip_gen_to_real, ar.compute_real_to_real, ar.compute_real_to_gen,
                ar.print_conf_mat, ar.norm_data)


if __name__ == '__main__':
  main()
