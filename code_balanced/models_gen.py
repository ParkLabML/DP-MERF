import torch as pt
import torch.nn as nn


# class FCGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, use_sigmoid=False, batch_norm=False):
#     super(FCGen, self).__init__()
#     self.fc1 = nn.Linear(d_code, d_hid[0])
#     self.fc2 = nn.Linear(d_hid[0], d_hid[1])
#     self.fc3 = nn.Linear(d_hid[1], d_enc)
#     self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
#     self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
#     self.use_sigmoid = use_sigmoid
#     self.d_code = d_code
#
#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.bn1(x) if self.bn1 is not None else x
#     x = self.fc2(self.relu(x))
#     x = self.bn2(x) if self.bn2 is not None else x
#     x = self.fc3(self.relu(x))
#     if self.use_sigmoid:
#       x = self.sigmoid(x)
#     return x
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)


# class FCGenBig(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, use_sigmoid=False, batch_norm=False):
#     super(FCGenBig, self).__init__()
#     self.fc1 = nn.Linear(d_code, d_hid[0])
#     self.fc2 = nn.Linear(d_hid[0], d_hid[1])
#     self.fc3 = nn.Linear(d_hid[1], d_hid[2])
#     self.fc4 = nn.Linear(d_hid[2], d_hid[3])
#     self.fc5 = nn.Linear(d_hid[3], d_enc)
#     self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
#     self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
#     self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
#     self.bn4 = nn.BatchNorm1d(d_hid[3]) if batch_norm else None
#     print(self.bn1)
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
#     self.use_sigmoid = use_sigmoid
#     self.d_code = d_code
#
#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.bn1(x) if self.bn1 is not None else x
#     x = self.fc2(self.relu(x))
#     x = self.bn2(x) if self.bn2 is not None else x
#     x = self.fc3(self.relu(x))
#     x = self.bn3(x) if self.bn3 is not None else x
#     x = self.fc4(self.relu(x))
#     x = self.bn4(x) if self.bn4 is not None else x
#     x = self.fc5(self.relu(x))
#     if self.use_sigmoid:
#       x = self.sigmoid(x)
#     return x
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)


# class FCLabelGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, n_labels=10, use_sigmoid=False, batch_norm=True):
#     super(FCLabelGen, self).__init__()
#     self.fc1 = nn.Linear(d_code, d_hid[0])
#     self.fc2 = nn.Linear(d_hid[0], d_hid[1])
#     self.data_layer = nn.Linear(d_hid[1], d_enc)
#     self.label_layer = nn.Linear(d_hid[1], n_labels)
#     self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
#     self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
#     self.use_sigmoid = use_sigmoid
#     self.softmax = nn.Softmax(dim=1)
#     self.d_code = d_code
#
#   def forward(self, x):
#     x = self.fc1(x)
#     x = self.bn1(x) if self.bn1 is not None else x
#     x = self.fc2(self.relu(x))
#     x = self.bn2(x) if self.bn1 is not None else x
#     x = self.relu(x)
#     x_data = self.data_layer(x)
#     x_labels = self.softmax(self.label_layer(x))
#     if self.use_sigmoid:
#       x_data = self.sigmoid(x_data)
#     return x_data, x_labels
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)


class FCCondGen(nn.Module):
  def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True):
    super(FCCondGen, self).__init__()
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(d_hid) < 5

    self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
    self.fc2 = nn.Linear(d_hid[0], d_hid[1])

    self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
    if len(d_hid) == 2:
      self.fc3 = nn.Linear(d_hid[1], d_out)
    elif len(d_hid) == 3:
      self.fc3 = nn.Linear(d_hid[1], d_hid[2])
      self.fc4 = nn.Linear(d_hid[2], d_out)
      self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
    elif len(d_hid) == 4:
      self.fc3 = nn.Linear(d_hid[1], d_hid[2])
      self.fc4 = nn.Linear(d_hid[2], d_hid[3])
      self.fc5 = nn.Linear(d_hid[3], d_out)
      self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
      self.bn4 = nn.BatchNorm1d(d_hid[3]) if batch_norm else None

    self.use_bn = batch_norm
    self.n_layers = len(d_hid)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.use_bn else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.use_bn else x
    x = self.fc3(self.relu(x))
    if self.n_layers > 2:
      x = self.bn3(x) if self.use_bn else x
      x = self.fc4(self.relu(x))
      if self.n_layers > 3:
        x = self.bn4(x) if self.use_bn else x
        x = self.fc5(self.relu(x))

    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class ConvCondGen(nn.Module):
  def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
    super(ConvCondGen, self).__init__()
    self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
    self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
    d_hid = [int(k) for k in d_hid.split(',')]
    assert len(self.nc) == 3 and len(self.ks) == 2
    self.hw = 7  # image height and width before upsampling
    self.reshape_size = self.nc[0]*self.hw**2
    self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
    self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
    self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
    self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
    self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.bn2 is not None else x
    # print(x.shape)
    x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.conv2(x)
    x = x.reshape(x.shape[0], -1)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class ConvCondGenSVHN(nn.Module):
  def __init__(self, d_code, fc_spec, n_labels, nc_str, ks_str, use_sigmoid=False, batch_norm=True):
    super(ConvCondGenSVHN, self).__init__()
    self.nc = [int(k) for k in nc_str.split(',')] + [3]  # number of channels
    self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
    fc_spec = [int(k) for k in fc_spec.split(',')]
    assert len(self.nc) == 4 and len(self.ks) == 3
    self.hw = 4  # image height and width before upsampling
    self.reshape_size = self.nc[0]*self.hw**2
    self.fc1 = nn.Linear(d_code + n_labels, fc_spec[0])
    self.fc2 = nn.Linear(fc_spec[0], self.reshape_size)
    self.bn1 = nn.BatchNorm1d(fc_spec[0]) if batch_norm else None
    self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
    self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
    self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
    self.conv3 = nn.Conv2d(self.nc[2], self.nc[3], kernel_size=self.ks[2], stride=1, padding=(self.ks[2] - 1) // 2)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.bn2 is not None else x
    # print(x.shape)
    x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.relu(self.conv2(x))
    x = self.upsamp(x)
    x = self.conv3(x)
    x = x.reshape(x.shape[0], -1)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device, return_labels=True, labels=None):
    if labels is None:  # sample labels
      labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    code = pt.randn(batch_size, self.d_code, device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code


class ConvGenSVHN(nn.Module):
  def __init__(self, d_code, fc_spec, n_labels, nc_str, ks_str, use_sigmoid=False, batch_norm=True):
    super(ConvGenSVHN, self).__init__()
    self.nc = [int(k) for k in nc_str.split(',')] + [3]  # number of channels
    self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
    assert len(self.nc) == 4 and len(self.ks) == 3
    self.hw = 4  # image height and width before upsampling
    self.reshape_size = self.nc[0]*self.hw**2
    fc_spec = [d_code] + [int(k) for k in fc_spec.split(',')] + [self.reshape_size]
    # print(fc_spec)
    self.fc1 = nn.Linear(fc_spec[0], fc_spec[1])
    self.bn1 = nn.BatchNorm1d(fc_spec[1]) if batch_norm else None
    self.fc2 = nn.Linear(fc_spec[1], fc_spec[2])
    self.bn2 = nn.BatchNorm1d(fc_spec[2]) if batch_norm else None
    if len(fc_spec) == 4:
      self.fc3 = nn.Linear(fc_spec[2], fc_spec[3])
      self.bn3 = nn.BatchNorm1d(fc_spec[3]) if batch_norm else None
    else:
      self.fc3, self.bn3 = None, None
    self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
    self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
    self.conv3 = nn.Conv2d(self.nc[2], self.nc[3], kernel_size=self.ks[2], stride=1, padding=(self.ks[2]-1) // 2)
    self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x) if self.bn1 is not None else x
    x = self.fc2(self.relu(x))
    x = self.bn2(x) if self.bn2 is not None else x

    x = self.fc3(self.relu(x)) if self.fc3 is not None else x
    x = self.bn3(x) if self.bn3 is not None else x

    x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
    x = self.upsamp(x)
    x = self.relu(self.conv1(x))
    x = self.upsamp(x)
    x = self.relu(self.conv2(x))
    x = self.upsamp(x)
    x = self.conv3(x)
    x = x.reshape(x.shape[0], -1)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x

  def get_code(self, batch_size, device):
    return pt.randn(batch_size, self.d_code, device=device)


# class FCGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc):
#     super(FCGen, self).__init__()
#     layer_spec = [d_code] + list(d_hid) + [d_enc]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     # self.tanh = nn.Tanh()
#     self.d_code = d_code
#
#   def forward(self, x):
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#     return x
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)


# class FCLabelGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, n_labels=10):
#     super(FCLabelGen, self).__init__()
#     layer_spec = [d_code] + list(d_hid)
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.data_layer = nn.Linear(layer_spec[-1], d_enc)
#     self.label_layer = nn.Linear(layer_spec[-1], n_labels)
#     self.relu = nn.ReLU()
#     self.softmax = nn.Softmax(dim=1)
#     self.d_code = d_code
#
#   def forward(self, x):
#     for layer in self.fc_layers:
#       x = self.relu(layer(x))
#     x_data = self.data_layer(x)
#     x_labels = self.softmax(self.label_layer(x))
#     return x_data, x_labels
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)
#
#
# class FCCondGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, n_labels):
#     super(FCCondGen, self).__init__()
#     layer_spec = [d_code + n_labels] + list(d_hid) + [d_enc]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     self.d_code = d_code
#     self.n_labels = n_labels
#
#   def forward(self, x):
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#     return x
#
#   def get_code(self, batch_size, device, return_labels=True):
#     code = pt.randn(batch_size, self.d_code, device=device)
#     sampled_labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
#     gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
#     gen_one_hots.scatter_(1, sampled_labels, 1)
#     code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
#     # print(code.shape)
#     if return_labels:
#       return code, gen_one_hots
#     else:
#       return code
