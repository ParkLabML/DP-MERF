"""
copied with minor changes from
https://github.com/reihaneh-torkzadehmahani/DP-CGAN/blob/master/DP_CGAN/dp_conditional_gan_mnist/DP_CGAN_RdpAcc.py
"""


# Import the requiered python packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import argparse


from dp_cgan_accounting.dp_sgd.dp_optimizer import dp_optimizer
from dp_cgan_accounting.dp_sgd.dp_optimizer import sanitizer
from dp_cgan_accounting.dp_sgd.dp_optimizer import utils
from dp_cgan_accounting.privacy_accountant.tf import accountant
from dp_cgan_accounting.analysis.rdp_accountant import compute_rdp
from dp_cgan_accounting.analysis.rdp_accountant import get_privacy_spent

try:
    from synth_data_2d import make_data_from_specstring, string_to_specs, plot_data
except ImportError:
    print('importing through relative path')
    # Import required Differential Privacy packages
    baseDir = "../code_balanced/"
    sys.path.append(baseDir)

    from synth_data_2d import make_data_from_specstring, string_to_specs, plot_data

tf.compat.v1.disable_eager_execution()


def xavier_init(size):
    """ Xavier Function to keep the scale of the gradients roughly the same
        in all the layers.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.compat.v1.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    """ Function to generate uniform prior for G(z)
    """
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y, theta_g):
    g_w1 = theta_g[0]
    g_w2 = theta_g[1]
    g_b1 = theta_g[2]
    g_b2 = theta_g[3]

    """ Function to build the generator network
    """
    inputs = tf.concat(axis=1, values=[z, y])
    g_h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)
    g_log_prob = tf.matmul(g_h1, g_w2) + g_b2
    # g_prob = tf.nn.sigmoid(g_log_prob)
    g_prob = g_log_prob
    return g_prob


def discriminator(x, y, theta_d):
    """ Function to build the discriminator network
    """
    d_w1 = theta_d[0]
    d_w2 = theta_d[1]
    d_b1 = theta_d[2]
    d_b2 = theta_d[3]

    inputs = tf.concat(axis=1, values=[x, y])
    d_h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
    d_logit = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_logit)

    return d_prob, d_logit


def plot(samples):
    """ Function to plot the generated images
    """
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(10, 1)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        # plt.show()
    return fig


def del_all_flags(FLAGS):
    """ Function to delete all flags before declare
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def compute_epsilon(batch_size, steps, sigma):
    """Computes epsilon value for given hyperparameters."""
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / 60000
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=sigma,
                      steps=steps,
                      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def batch_dataset(data, labels, batch_size):
    n_data = len(data)
    start_idx = n_data + 1  # trigger shuffle on first call
    while True:

        if start_idx + batch_size > n_data:
            rand_perm = np.random.permutation(n_data)
            data = data[rand_perm]
            labels = labels[rand_perm]
            start_idx = 0

        old_idx = start_idx
        start_idx = start_idx+batch_size
        yield data[old_idx:start_idx], labels[old_idx:start_idx]




def runTensorFlow(sigma, clipping_value, batch_size, epsilon, delta, iteration, log_name,
                  spec_string, h_dim, z_dim, start_lr, end_lr, lr_saturate_epochs, num_training_steps, base_save_dir):

    # num_training_images = 60000
    # Initializations for a two-layer discriminator network
    # code_balanced = input_data.read_data_sets(baseDir + "our_dp_conditional_gan_mnist/mnist_dataset", one_hot=True)
    # if dataset_key == 'digits':
    #     code_balanced = input_data.read_data_sets("data/MNIST/raw", one_hot=True)
    # elif dataset_key == 'fashion':
    #     print('using FashionMNIST')
    #     code_balanced = input_data.read_data_sets("../../data/FashionMNIST/raw", one_hot=True)
    # else:
    #     raise ValueError

    data_samples, label_samples, eval_func, class_centers = make_data_from_specstring(spec_string)

    n_samples = data_samples.shape[0]
    n_labels = class_centers.shape[0]

    labels = np.zeros((n_samples, n_labels))
    labels[np.arange(label_samples.size), label_samples] = 1  # convert to one-hot

    batch_generator = batch_dataset(data_samples, labels, batch_size)

    x_dim = data_samples.shape[1]
    y_dim = labels.shape[1]
    x_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim])
    y_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, y_dim])

    d_w1 = tf.Variable(xavier_init([x_dim + y_dim, h_dim]))
    d_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    d_w2 = tf.Variable(xavier_init([h_dim, 1]))
    d_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_d = [d_w1, d_w2, d_b1, d_b2]

    # Initializations for a two-layer genrator network
    z_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])
    g_w1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
    g_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    g_w2 = tf.Variable(xavier_init([h_dim, x_dim]))
    g_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
    theta_g = [g_w1, g_w2, g_b1, g_b2]


    priv_accountant = accountant.GaussianMomentsAccountant(data_samples.shape[0])

    # Sanitizer
    # batch_size = FLAGS.batch_size
    # clipping_value = FLAGS.default_gradient_l2norm_bound
    # clipping_value = tf.placeholder(tf.float32)
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant, [clipping_value / batch_size, True])

    # Instantiate the Generator Network
    g_sample = generator(z_pl, y_pl, theta_g)

    # Instantiate the Discriminator Network
    d_real, d_logit_real = discriminator(x_pl, y_pl, theta_d)
    d_fake, d_logit_fake = discriminator(g_sample, y_pl, theta_d)

    # Discriminator loss for real data
    d_loss_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real))
    d_loss_real = tf.reduce_mean(d_loss_real_ce, [0])
    # Discriminator loss for fake data
    d_loss_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake))
    d_loss_fake = tf.reduce_mean(d_loss_fake_ce, [0])

    # Generator loss
    g_loss_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake))
    g_loss = tf.reduce_mean(g_loss_ce, [0])

    # ------------------------------------------------------------------------------
    """
    minimize_ours :
            Our method (Clipping the gradients of loss on real data and making
            them noisy + Clipping the gradients of loss on fake data) is
            implemented in this function .
            It can be found in the following directory:
            differential_privacy/dp_sgd/dp_optimizer/dp_optimizer.py'
    """
    lr_pl = tf.compat.v1.placeholder(tf.float32)
    # sigma = FLAGS.sigma
    # Generator optimizer
    g_solver = tf.compat.v1.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)
    # Discriminator Optimizer

    if epsilon > 0:
        d_optim = dp_optimizer.DPGradientDescentOptimizer(lr_pl, [None, None], gaussian_sanitizer, sigma=sigma,
                                                          batches_per_lot=1)

        d_solver = d_optim.minimize_ours(d_loss_real, d_loss_fake, var_list=theta_d)
    else:
        d_optim = tf.compat.v1.train.MomentumOptimizer(lr_pl, momentum=0.)
        d_solver = d_optim.minimize(d_loss_real + d_loss_fake, var_list=theta_d)
    # ------------------------------------------------------------------------------

    # Set output directory
    result_dir = baseDir + "out/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/synth2d_run_{}_bs_{}_s_{}_c_{}_d_{}_e_{}".format(iteration, batch_size, sigma,
                                                                                  clipping_value, delta, str(epsilon))

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    target_eps = [float(s) for s in str(epsilon).split(",")]
    max_target_eps = max(target_eps)

    # gpu_options = tf.GPUOptions(visible_device_list="0, 1")
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")

    # Main Session
    with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        step = 0

        # Is true when the spent privacy budget exceeds the target budget
        should_terminate = False

        # Main loop
        while step <= num_training_steps and should_terminate is False:

            epoch = step
            curr_lr = utils.VaryRate(start_lr, end_lr, lr_saturate_epochs, epoch)

            if sigma > 0:
                eps = compute_epsilon(batch_size, (step + 1), sigma * clipping_value)
            else:
                eps = 0

            # x_mb, y_mb = code_balanced.train.next_batch(batch_size, shuffle=True)
            x_mb, y_mb = next(batch_generator)
            z_sample = sample_z(batch_size, z_dim)

            # Update the discriminator network
            _, d_loss_real_curr, d_loss_fake_curr = sess.run([d_solver, d_loss_real, d_loss_fake],
                                                             feed_dict={x_pl: x_mb, z_pl: z_sample,
                                                                        y_pl: y_mb, lr_pl: curr_lr})

            # Update the generator network
            _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z_pl: z_sample, y_pl: y_mb, lr_pl: curr_lr})

            # print(f'checking eps: eps: {eps}/{max_target_eps}')
            if (max_target_eps > 0. and eps >= max_target_eps) or step >= num_training_steps:
                print(f"TERMINATE!!!!")
                print("Termination Step : " + str(step))

                n_class = np.zeros(n_labels) + n_samples // n_labels  # assume balanced data

                gen_labels = np.zeros(shape=[n_samples, n_labels])
                image_cntr = 0
                for class_cntr in np.arange(len(n_class)):
                    for cntr in np.arange(n_class[class_cntr]):
                        gen_labels[image_cntr, class_cntr] = 1
                        image_cntr += 1

                z_sample = sample_z(n_samples, z_dim)

                images = sess.run(g_sample, feed_dict={z_pl: z_sample, y_pl: gen_labels})

                gen_labels = np.argmax(gen_labels, axis=1)  # one-hot to scalar
                print(f'saving genereated data of shape {images.shape} and {gen_labels.shape}')
                save_dir = os.path.join(base_save_dir, f'dp-cgan-synth-2d-{spec_string}-eps{max_target_eps}', log_name)
                os.makedirs(save_dir, exist_ok=True)
                save_str = os.path.join(save_dir, 'samples.npz')
                np.savez(save_str, data=images, labels=gen_labels)
                print(f'samples saved at: {save_str}')

                break  # out of while loop, ending the function

            step = step + 1
    return save_dir


def test_data(save_dir, data_spec_str):
    samples = np.load(os.path.join(save_dir, 'samples.npz'))
    gen_x = samples['data']
    gen_y = samples['labels']
    data_samples, label_samples, eval_func, class_centers = make_data_from_specstring(data_spec_str)
    plot_data(gen_x, gen_y, os.path.join(save_dir, 'gen_data'))
    plot_data(gen_x, gen_y, os.path.join(save_dir, 'gen_data_sub0.1'), subsample=0.1)
    plot_data(gen_x, gen_y, os.path.join(save_dir, 'gen_data_centered'), center_frame=True)
    print(f'gen samples eval score: {eval_func(gen_x, gen_y)}')


def main():
    # sigma_clipping_list = [[1.12, 1.1]]
    # sigma_clipping_list = [[0.1, 1.1]]
    # batchSizeList = [600]
    # epsilon = 9.6
    # epsilon = 1e10
    # delta = 1e-5

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-name', type=str, default=None)  # name run for saving
    # parser.add_argument('--data', type=str, default='digits')  # options are digits and fashion
    parser.add_argument('--synth-spec-string', type=str, default='norm_k5_n100000_row5_col5_noise0.2', help='')
    parser.add_argument('--base-save-dir', type=str, default='logs/')
    parser.add_argument('--epsilon', type=float, default=1.0, help='privacy epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy delta')
    parser.add_argument('--noise', type=float, default=2.0, help='privacy noise sigma')
    parser.add_argument('--batch-size', '-bs', type=int, default=500)
    parser.add_argument('--clip', type=float, default=1.1, help='norm clip')
    parser.add_argument('--h-dim', type=int, default=128)
    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--start-lr', type=float, default=0.1)
    parser.add_argument('--end-lr', type=float, default=0.052)
    parser.add_argument('--lr-saturate-epochs', type=int, default=10_000)

    parser.add_argument('--num-training-steps', type=int, default=100_000)
    ar = parser.parse_args()
    assert ar.log_name is not None

    save_dir = runTensorFlow(ar.noise, ar.clip, ar.batch_size, ar.epsilon, ar.delta, 1, ar.log_name,
                             ar.synth_spec_string, ar.h_dim, ar.z_dim, ar.start_lr, ar.end_lr, ar.lr_saturate_epochs,
                             ar.num_training_steps, ar.base_save_dir)

    test_data(save_dir, ar.synth_spec_string)


if __name__ == '__main__':
    main()


